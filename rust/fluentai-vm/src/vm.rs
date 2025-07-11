//! High-performance stack-based virtual machine
use  fluentai_bytecode::{Bytecode, Instruction, Opcode};
use  crate::cow_globals::CowGlobals;
use  crate::debug::{DebugConfig, StepMode, VMDebugEvent};
use  crate::error::{value_type_name, StackFrame, StackTrace, VMError, VMResult};
use  crate::gc::{GarbageCollector, GcConfig, GcScope};
#[cfg(feature = "jit")]
use  crate::jit_integration::{JitConfig, JitManager};
use  crate::safety::{checked_ops, ActorId, ChannelId, IdGenerator, PromiseId, ResourceLimits};
use  crate::security::{SecurityManager, SecurityPolicy};
use  fluentai_core::ast::{NodeId, UsageStatistics};
use  fluentai_core::value::Value;
use  fluentai_effects::{runtime::EffectRuntime, EffectContext};
use  fluentai_modules::{ModuleLoader, ModuleResolver};
use  fluentai_stdlib::value::Value as StdlibValue;
use  fluentai_stdlib::{init_stdlib, StdlibRegistry};
use  rustc_hash::FxHashMap;
use  std::sync::{Arc, RwLock};
use  std::time::Instant;
use  tokio::sync::{mpsc, oneshot};

const STACK_SIZE: usize = 10_000;
const MAX_PRESERVED_LOCALS: usize = 1000;
const FINALLY_NORMAL_MARKER: &str = "__finally_normal__";
const FINALLY_EXCEPTION_MARKER: &str = "__finally_exception__";

/// Bit masks for MakeClosure instruction unpacking
const MAKECLOSURE_CHUNK_ID_SHIFT: u32 = 16;
const MAKECLOSURE_CAPTURE_COUNT_MASK: u32 = 0xFFFF;

/// Actor state and message handling
pub struct Actor {
    /// Current state of the actor
    state: Value,
    /// Handler function that processes messages
    handler: Value,
    /// Mailbox for incoming messages
    mailbox: mpsc::Receiver<Value>,
    /// Sender for the mailbox
    sender: mpsc::Sender<Value>,
}

pub struct CallFrame {
    pub chunk_id: usize,
    pub ip: usize,
    pub stack_base: usize,
    pub env: Vec<Value>, // Captured environment for closures
    #[allow(dead_code)]
    pub start_time: Option<Instant>, // Track when this frame started executing
}

/// Tracks usage statistics for nodes during execution
pub struct UsageTracker {
    /// Map from chunk_id to NodeId for tracking
    chunk_to_node: FxHashMap<usize, NodeId>,
    /// Accumulated statistics per node
    stats: FxHashMap<NodeId, UsageStatistics>,
    /// Execution time tracking
    execution_times: FxHashMap<NodeId, Vec<u64>>,
}

impl UsageTracker {
    pub fn new() -> Self {
        Self {
            chunk_to_node: FxHashMap::default(),
            stats: FxHashMap::default(),
            execution_times: FxHashMap::default(),
        }
    }

    /// Register a chunk ID to node ID mapping
    pub fn register_chunk(&mut self, chunk_id: usize, node_id: NodeId) {
        self.chunk_to_node.insert(chunk_id, node_id);
    }

    /// Record execution of a chunk
    pub fn record_execution(&mut self, chunk_id: usize, execution_time_ns: u64) {
        if let Some(&node_id) = self.chunk_to_node.get(&chunk_id) {
            let stats = self.stats.entry(node_id).or_default();
            stats.execution_count += 1;

            // Update average execution time
            let times = self.execution_times.entry(node_id).or_default();
            times.push(execution_time_ns);

            // Keep last 100 samples for moving average
            if times.len() > 100 {
                times.remove(0);
            }

            stats.avg_execution_time_ns = times.iter().sum::<u64>() / times.len() as u64;

            // Mark as hot path if executed frequently
            if stats.execution_count > 1000 {
                stats.is_hot_path = true;
            }
        }
    }

    /// Record an error for a chunk
    pub fn record_error(&mut self, chunk_id: usize) {
        if let Some(&node_id) = self.chunk_to_node.get(&chunk_id) {
            let stats = self.stats.entry(node_id).or_default();
            stats.error_count += 1;
        }
    }

    /// Get statistics for a node
    pub fn get_stats(&self, node_id: NodeId) -> Option<&UsageStatistics> {
        self.stats.get(&node_id)
    }

    /// Get all statistics
    pub fn get_all_stats(&self) -> &FxHashMap<NodeId, UsageStatistics> {
        &self.stats
    }
    
    /// Get usage statistics for a specific chunk
    pub fn get_stats_for_chunk(&self, chunk_id: usize) -> Option<UsageStatistics> {
        let node_id = self.chunk_to_node.get(&chunk_id)?;
        self.stats.get(node_id).cloned()
    }
}

pub struct VM {
    bytecode: Arc<Bytecode>,
    stack: Vec<Value>,
    call_stack: Vec<CallFrame>,
    globals: CowGlobals,
    trace: bool,
    effect_context: Arc<EffectContext>,
    effect_runtime: Arc<EffectRuntime>,
    // Async support with typed IDs
    id_generator: IdGenerator,
    promises: FxHashMap<PromiseId, oneshot::Receiver<VMResult<Value>>>,
    channels: FxHashMap<ChannelId, (mpsc::Sender<Value>, mpsc::Receiver<Value>)>,
    // Actor support
    actors: FxHashMap<ActorId, Actor>,
    // Mutable cells
    cells: Vec<Value>,
    // Standard library
    stdlib: StdlibRegistry,
    // Module system
    module_loader: ModuleLoader,
    #[allow(dead_code)]
    module_resolver: ModuleResolver,
    loaded_modules: FxHashMap<String, Value>, // Cache of loaded modules
    current_module: Option<String>,           // Name of currently executing module
    module_stack: Vec<String>,                // Stack of module names for nested module execution
    // Debug support
    debug_config: DebugConfig,
    instruction_count: u64,
    // Resource limits
    resource_limits: ResourceLimits,
    // Security manager
    security_manager: Option<Arc<SecurityManager>>,
    // Garbage collector
    gc: Option<Arc<GarbageCollector>>,
    // Usage tracking for context memory
    usage_tracker: Option<Arc<RwLock<UsageTracker>>>,
    // Effect handler stack
    handler_stack: Vec<HandlerFrame>,
    // Error handler stack for try-catch-finally
    error_handler_stack: Vec<ErrorHandler>,
    // Finally block state storage
    finally_states: Vec<FinallyState>,
    // Current actor context (when executing actor handlers)
    current_actor: Option<ActorId>,
    // Current message being processed by actor (for ActorReceive opcode)
    current_actor_message: Option<Value>,
    // JIT compilation manager
    #[cfg(feature = "jit")]
    jit_manager: JitManager,
}

/// Handler frame for tracking active effect handlers
#[derive(Clone)]
pub struct HandlerFrame {
    /// Map from effect type + operation to handler function value
    handlers: FxHashMap<(String, Option<String>), Value>,
    /// Continuation point - where to return after handler execution
    _return_ip: usize,
    /// Stack depth when handler was installed
    _stack_depth: usize,
}

/// Error handler for try-catch-finally blocks
#[derive(Clone)]
struct ErrorHandler {
    /// Catch handler IP (where to jump on error)
    catch_ip: usize,
    /// Finally handler IP (optional)
    finally_ip: Option<usize>,
    /// Stack depth when handler was installed
    stack_depth: usize,
    /// Call frame index
    call_frame: usize,
    /// Number of local variables at the handler's scope
    locals_count: usize,
}

/// State saved during finally block execution
#[derive(Clone)]
struct FinallyState {
    /// The value to restore after finally (result or error)
    value: Value,
    /// Marker indicating normal or exception path
    marker: Value,
}

impl VM {
    pub fn new(bytecode: Bytecode) -> Self {
        Self::with_shared_bytecode(Arc::new(bytecode))
    }
    
    pub fn with_shared_bytecode(bytecode: Arc<Bytecode>) -> Self {
        Self {
            bytecode,
            stack: Vec::with_capacity(STACK_SIZE),
            call_stack: Vec::new(),
            globals: CowGlobals::new(),
            trace: false,
            effect_context: Arc::new(EffectContext::default()),
            effect_runtime: Arc::new(EffectRuntime::default()),
            id_generator: IdGenerator::new(),
            promises: FxHashMap::default(),
            channels: FxHashMap::default(),
            actors: FxHashMap::default(),
            cells: Vec::new(),
            stdlib: init_stdlib(),
            module_loader: ModuleLoader::new(fluentai_modules::ModuleConfig::default()),
            module_resolver: ModuleResolver::new(ModuleLoader::new(
                fluentai_modules::ModuleConfig::default(),
            )),
            loaded_modules: FxHashMap::default(),
            current_module: None,
            module_stack: Vec::new(),
            debug_config: DebugConfig::default(),
            instruction_count: 0,
            resource_limits: ResourceLimits::default(),
            security_manager: None,
            gc: None,
            usage_tracker: None,
            handler_stack: Vec::new(),
            error_handler_stack: Vec::new(),
            finally_states: Vec::new(),
            current_actor: None,
            current_actor_message: None,
            #[cfg(feature = "jit")]
            jit_manager: JitManager::new(JitConfig::default()),
        }
    }

    pub fn enable_trace(&mut self) {
        self.trace = true;
    }

    /// Reset VM state for reuse while keeping expensive initializations
    pub fn reset(&mut self) {
        // Clear runtime state
        self.stack.clear();
        self.call_stack.clear();
        self.globals.clear();
        self.promises.clear();
        self.channels.clear();
        self.cells.clear();
        self.instruction_count = 0;
        self.handler_stack.clear();
        self.error_handler_stack.clear();
        self.finally_states.clear();

        // Keep these expensive initializations:
        // - self.stdlib (258 functions)
        // - self.module_loader
        // - self.module_resolver
        // - self.effect_context
        // - self.effect_runtime
        // - self.id_generator
        // - self.resource_limits
        // - self.security_manager
        // - self.gc
        // - self.usage_tracker

        // Clear module state but keep the loader
        self.loaded_modules.clear();
        self.current_module = None;
        self.module_stack.clear();
    }

    /// Enable usage tracking
    pub fn enable_usage_tracking(&mut self) {
        self.usage_tracker = Some(Arc::new(RwLock::new(UsageTracker::new())));
    }

    /// Get usage tracker
    pub fn usage_tracker(&self) -> Option<Arc<RwLock<UsageTracker>>> {
        self.usage_tracker.clone()
    }

    /// Register a chunk to node mapping for usage tracking
    pub fn register_chunk_mapping(&mut self, chunk_id: usize, node_id: NodeId) {
        if let Some(tracker) = &self.usage_tracker {
            if let Ok(mut tracker) = tracker.write() {
                tracker.register_chunk(chunk_id, node_id);
            }
        }
    }

    pub fn set_effect_runtime(&mut self, runtime: Arc<EffectRuntime>) {
        self.effect_runtime = runtime;
    }

    pub fn get_effect_context(&self) -> Arc<EffectContext> {
        self.effect_context.clone()
    }

    pub fn set_effect_context(&mut self, context: Arc<EffectContext>) {
        self.effect_context = context;
    }

    pub fn set_stdlib_registry(&mut self, registry: StdlibRegistry) {
        self.stdlib = registry;
    }

    pub fn set_module_loader(&mut self, loader: ModuleLoader) {
        self.module_loader = loader;
    }

    pub fn set_security_manager(&mut self, manager: Arc<SecurityManager>) {
        self.security_manager = Some(manager);
    }

    pub fn with_sandbox_security(&mut self) -> &mut Self {
        self.security_manager = Some(Arc::new(SecurityManager::sandbox()));
        self
    }

    pub fn with_security_policy(&mut self, policy: SecurityPolicy) -> &mut Self {
        self.security_manager = Some(Arc::new(SecurityManager::new(policy)));
        self
    }

    /// Enable garbage collection with default configuration
    pub fn with_gc(&mut self) -> &mut Self {
        self.gc = Some(Arc::new(GarbageCollector::new(GcConfig::default())));
        self
    }

    /// Enable garbage collection with custom configuration
    pub fn with_gc_config(&mut self, config: GcConfig) -> &mut Self {
        self.gc = Some(Arc::new(GarbageCollector::new(config)));
        self
    }

    /// Get the garbage collector if enabled
    pub fn gc(&self) -> Option<&Arc<GarbageCollector>> {
        self.gc.as_ref()
    }

    /// Allocate a value using GC if enabled
    pub fn gc_alloc(&self, value: Value) -> VMResult<Value> {
        if let Some(ref gc) = self.gc {
            let handle = gc.allocate(value)?;
            // Store handle ID in a special GC value variant
            Ok(Value::GcHandle(Arc::new(handle)))
        } else {
            // If GC is not enabled, return the value as-is
            Ok(value)
        }
    }

    /// Create a GC scope for temporary allocations
    pub fn gc_scope<F, R>(&self, f: F) -> VMResult<R>
    where
        F: FnOnce(&mut GcScope) -> VMResult<R>,
    {
        if let Some(ref gc) = self.gc {
            let mut scope = GcScope::new(gc);
            f(&mut scope)
        } else {
            Err(VMError::RuntimeError {
                message: "GC not enabled".to_string(),
                stack_trace: None,
            })
        }
    }

    /// Manually trigger garbage collection
    pub fn gc_collect(&self) -> VMResult<()> {
        if let Some(ref gc) = self.gc {
            gc.collect().map_err(|e| VMError::RuntimeError {
                message: format!("GC error: {}", e),
                stack_trace: None,
            })
        } else {
            Ok(()) // No-op if GC not enabled
        }
    }

    pub fn set_debug_config(&mut self, config: DebugConfig) {
        self.debug_config = config;
    }

    pub fn get_debug_config(&self) -> &DebugConfig {
        &self.debug_config
    }

    pub fn get_debug_config_mut(&mut self) -> &mut DebugConfig {
        &mut self.debug_config
    }

    /// Get current VM state for debugging
    pub fn get_stack(&self) -> &[Value] {
        &self.stack
    }

    pub fn get_globals(&self) -> FxHashMap<String, Value> {
        self.globals.as_map()
    }

    pub fn get_call_stack_depth(&self) -> usize {
        self.call_stack.len()
    }

    /// Get usage statistics for a specific node
    pub fn get_usage_stats(&self, node_id: NodeId) -> Option<UsageStatistics> {
        self.usage_tracker
            .as_ref()?
            .read()
            .ok()?
            .get_stats(node_id)
            .cloned()
    }

    /// Get all usage statistics
    pub fn get_all_usage_stats(&self) -> Option<FxHashMap<NodeId, UsageStatistics>> {
        self.usage_tracker
            .as_ref()?
            .read()
            .ok()
            .map(|tracker| tracker.get_all_stats().clone())
    }

    pub fn run(&mut self) -> VMResult<Value> {
        self.call_stack.push(CallFrame {
            chunk_id: self.bytecode.main_chunk,
            ip: 0,
            stack_base: 0,
            env: Vec::new(),
            start_time: self.usage_tracker.as_ref().map(|_| Instant::now()),
        });

        let result = self.run_inner();

        // Track error if one occurred
        if result.is_err() {
            if let Some(tracker) = &self.usage_tracker {
                if let Some(frame) = self.call_stack.last() {
                    if let Ok(mut tracker_guard) = tracker.write() {
                        tracker_guard.record_error(frame.chunk_id);
                    }
                }
            }
        }

        result
    }

    /// Run until the VM completes and return the final value
    pub fn run_until_complete(&mut self) -> VMResult<Value> {
        // This is similar to run() but doesn't set up the initial call frame
        // as it's expected to be already set up by the caller
        self.run_inner()
    }
    
    fn run_inner(&mut self) -> VMResult<Value> {
        loop {
            let frame = self
                .call_stack
                .last()
                .ok_or_else(|| VMError::StackUnderflow {
                    operation: "get_current_frame".to_string(),
                    stack_size: self.call_stack.len(),
                    stack_trace: None,
                })?;
            let chunk_id = frame.chunk_id;
            let ip = frame.ip;

            if ip >= self.bytecode.chunks[chunk_id].instructions.len() {
                return Err(VMError::InvalidJumpTarget {
                    target: ip,
                    chunk_size: self.bytecode.chunks[chunk_id].instructions.len(),
                    stack_trace: None,
                });
            }

            let instruction = self.bytecode.chunks[chunk_id].instructions[ip].clone();

            // Check for breakpoints
            if self.debug_config.enabled && self.debug_config.should_break(ip) {
                self.debug_config
                    .send_event(VMDebugEvent::Breakpoint { pc: ip });
                self.debug_config.step_mode = StepMode::Step;
            }

            // Send pre-instruction debug event
            if self.debug_config.enabled {
                self.debug_config.send_event(VMDebugEvent::PreInstruction {
                    pc: ip,
                    instruction: instruction.clone(),
                    stack_size: self.stack.len(),
                });
            }

            if self.trace {
                eprintln!("Stack: {:?}", self.stack);
                eprintln!("Executing: {:?} at {}", instruction.opcode, ip);
            }

            // Security checks
            if let Some(ref security) = self.security_manager {
                security.context.track_instruction()?;

                // Check specific instruction security requirements
                match &instruction.opcode {
                    Opcode::Call => {
                        // Check call depth
                        if self.call_stack.len() >= self.resource_limits.max_call_depth {
                            return Err(VMError::CallStackOverflow {
                                current_depth: self.call_stack.len(),
                                max_depth: self.resource_limits.max_call_depth,
                                stack_trace: None,
                            });
                        }
                    }
                    _ => {}
                }
            }

            // Increment IP before execution (may be modified by jumps)
            self.call_stack.last_mut().unwrap().ip += 1;
            self.instruction_count += 1;

            match self.execute_instruction(&instruction, chunk_id)? {
                VMState::Continue => {
                    // Send post-instruction debug event
                    if self.debug_config.enabled {
                        let stack_top = self.stack.last().cloned();
                        self.debug_config.send_event(VMDebugEvent::PostInstruction {
                            pc: ip,
                            stack_size: self.stack.len(),
                            stack_top,
                        });
                    }
                }
                VMState::Return => {
                    if self.call_stack.len() == 1 {
                        // Main function returning
                        let result = self.stack.pop().ok_or_else(|| VMError::StackUnderflow {
                            operation: "main_return".to_string(),
                            stack_size: self.stack.len(),
                            stack_trace: None,
                        })?;

                        // Track main function execution time
                        if let Some(tracker) = &self.usage_tracker {
                            if let Some(frame) = self.call_stack.last() {
                                if let Some(start_time) = frame.start_time {
                                    let elapsed = start_time.elapsed().as_nanos() as u64;
                                    if let Ok(mut tracker_guard) = tracker.write() {
                                        tracker_guard.record_execution(frame.chunk_id, elapsed);
                                    }
                                }
                            }
                        }

                        if self.debug_config.enabled {
                            self.debug_config.send_event(VMDebugEvent::FunctionReturn {
                                value: result.clone(),
                                call_depth: self.call_stack.len(),
                            });
                        }
                        return Ok(result);
                    }
                    // Pop call frame and continue
                    self.call_stack.pop();
                    if self.debug_config.enabled {
                        let return_value = self.stack.last().cloned().unwrap_or(Value::Nil);
                        self.debug_config.send_event(VMDebugEvent::FunctionReturn {
                            value: return_value,
                            call_depth: self.call_stack.len(),
                        });
                    }
                }
                VMState::Halt => {
                    return self.stack.pop().ok_or_else(|| VMError::StackUnderflow {
                        operation: "halt".to_string(),
                        stack_size: self.stack.len(),
                        stack_trace: None,
                    });
                }
            }

            // Handle step mode
            if self.debug_config.enabled {
                match self.debug_config.step_mode {
                    StepMode::Step => {
                        // Pause after each instruction
                        self.debug_config.step_mode = StepMode::Run;
                        // In a real implementation, we'd wait for a continue signal here
                    }
                    StepMode::StepOver => {
                        // Continue until we're back at the same call depth
                        // Implementation would track the initial call depth
                    }
                    StepMode::StepOut => {
                        // Continue until we return from current function
                        // Implementation would track when we exit current frame
                    }
                    StepMode::Run => {
                        // Continue normally
                    }
                }
            }
        }
    }

    pub fn execute_instruction(
        &mut self,
        instruction: &Instruction,
        chunk_id: usize,
    ) -> VMResult<VMState> {
                use crate::opcode_handlers::{
                ArithmeticHandler, StackHandler, ControlFlowHandler,
                MemoryHandler, CollectionsHandler, ConcurrentHandler,
                EffectsHandler, LogicalHandler, OpcodeHandler
            };
            use Opcode::*;
            
            // Create handler instances
            let mut arithmetic_handler = ArithmeticHandler;
            let mut stack_handler = StackHandler;
            let mut control_flow_handler = ControlFlowHandler;
            let mut memory_handler = MemoryHandler;
            let mut collections_handler = CollectionsHandler;
            let mut concurrent_handler = ConcurrentHandler;
            let mut effects_handler = EffectsHandler;
            let mut logical_handler = LogicalHandler;
            
            // Dispatch to appropriate handler based on opcode category
            match instruction.opcode {
                // Stack operations - dispatched to StackHandler
                Push | Pop | PopN | Dup | Swap |
                PushInt0 | PushInt1 | PushInt2 | PushIntSmall |
                PushTrue | PushFalse | PushNil | PushConst => {
                    return stack_handler.execute(self, instruction, chunk_id);
                }
                
                // Arithmetic operations - dispatched to ArithmeticHandler
                Add | Sub | Mul | Div | Mod | Neg |
                AddInt | SubInt | MulInt | DivInt |
                AddFloat | SubFloat | MulFloat | DivFloat => {
                    return arithmetic_handler.execute(self, instruction, chunk_id);
                }
                
                // Logical operations - dispatched to LogicalHandler
                Eq | Ne | Lt | Le | Gt | Ge |
                LtInt | LeInt | GtInt | GeInt |
                And | Or | Not => {
                    return logical_handler.execute(self, instruction, chunk_id);
                }
                
                // Control flow operations - dispatched to ControlFlowHandler
                Jump | JumpIf | JumpIfNot | Call | TailCall | 
                Return | TailReturn | LoopStart | LoopEnd | Halt => {
                    return control_flow_handler.execute(self, instruction, chunk_id);
                }
                
                // Memory operations - dispatched to MemoryHandler
                Load | Store | LoadLocal | StoreLocal |
                LoadLocal0 | LoadLocal1 | LoadLocal2 | LoadLocal3 |
                StoreLocal0 | StoreLocal1 | StoreLocal2 | StoreLocal3 |
                LoadGlobal | StoreGlobal | DefineGlobal |
                LoadCaptured | LoadUpvalue | StoreUpvalue |
                MakeCell | LoadCell | StoreCell | UpdateLocal => {
                    return memory_handler.execute(self, instruction, chunk_id);
                }
                
                // Collection operations - dispatched to CollectionsHandler
                MakeList | ListGet | ListSet | ListHead | ListTail |
                ListCons | ListLen | ListEmpty |
                MakeMap | MapGet | MapSet => {
                    return collections_handler.execute(self, instruction, chunk_id);
                }
                
                // Concurrent operations - dispatched to ConcurrentHandler
                Spawn | Await | Channel | ChannelWithCapacity | MakeChannel |
                Send | Receive | CreateActor | MakeActor | ActorSend |
                ActorReceive | Become | TrySend | TryReceive | Select |
                PromiseNew | PromiseAll | PromiseRace | WithTimeout => {
                    return concurrent_handler.execute(self, instruction, chunk_id);
                }
                
                // Effect operations - dispatched to EffectsHandler
                Effect | EffectAsync | Perform | MakeHandler |
                InstallHandler | UninstallHandler | Resume |
                Try | Catch | Finally | EndFinally |
                PushHandler | PushFinally | PopHandler |
                TryStart | TryStartWithFinally | TryEnd | Throw |
                FinallyStart | FinallyEnd => {
                    return effects_handler.execute(self, instruction, chunk_id);
                }
                
                // Remaining opcodes that are handled directly
                // These could potentially be moved to new handlers in the future
                
                MakeFunc => {
                    let chunk_id = instruction.arg as usize;
                    // MakeFunc is used for functions with no free variables
                    // Functions with free variables use MakeClosure instead
                    let env = Vec::new();
                    let func = Value::Function { chunk_id, env };
                    self.push(func)?;
                }
                
                MakeClosure => {
                    // MakeClosure bit unpacking:
                    // The packed argument contains both the chunk ID and capture count
                    // Upper 16 bits: chunk_id - which bytecode chunk contains the function
                    // Lower 16 bits: capture_count - how many values to capture from stack
                    let packed = instruction.arg;
                    let chunk_id = (packed >> MAKECLOSURE_CHUNK_ID_SHIFT) as usize;
                    let capture_count = (packed & MAKECLOSURE_CAPTURE_COUNT_MASK) as usize;
                    // Pop captured values from stack
                    let mut env = Vec::with_capacity(capture_count);
                    for _ in 0..capture_count {
                        env.push(self.pop()?);
                    }
                    env.reverse(); // Restore original order
                    
                    let func = Value::Function { chunk_id, env };
                    self.push(func)?;
                }
                
                MakeFuture => {
                    // Pop the function and convert it to a future
                    let func = self.pop()?;
                    match func {
                        Value::Function { chunk_id, env } => {
                            let future = Value::Future { chunk_id, env };
                            self.push(future)?;
                        }
                        v => {
                            return Err(VMError::TypeError {
                                operation: "make_future".to_string(),
                                expected: "function".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                MakeEnv => {
                    // MakeEnv could be used for creating dynamic environments
                    // or implementing with-environment constructs
                    // Currently environments are managed through closures
                    // This is a no-op for now
                }
                
                PopEnv => {
                    // PopEnv would restore a previous environment
                    // Currently not used as environments are handled via closures
                    // This is a no-op for now
                }
                
                StrLen => {
                    let string = self.pop()?;
                    match string {
                        Value::String(s) => self.push(Value::Integer(s.len() as i64))?,
                        v => {
                            return Err(VMError::TypeError {
                                operation: "str_len".to_string(),
                                expected: "string".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                StrConcat => self.binary_op(|a, b| match (a, b) {
                    (Value::String(x), Value::String(y)) => Ok(Value::String(x + &y)),
                    (a, b) => Err(VMError::TypeError {
                        operation: "str_concat".to_string(),
                        expected: "string".to_string(),
                        got: format!("{} and {}", value_type_name(&a), value_type_name(&b)),
                        location: None,
                        stack_trace: None,
                    }),
                })?,
                
                StrUpper => {
                    let string = self.pop()?;
                    match string {
                        Value::String(s) => self.push(Value::String(s.to_uppercase()))?,
                        v => {
                            return Err(VMError::TypeError {
                                operation: "str_upper".to_string(),
                                expected: "string".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                StrLower => {
                    let string = self.pop()?;
                    match string {
                        Value::String(s) => self.push(Value::String(s.to_lowercase()))?,
                        v => {
                            return Err(VMError::TypeError {
                                operation: "str_lower".to_string(),
                                expected: "string".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                CellGet => {
                    let cell = self.pop()?;
                    match cell {
                        Value::Cell(idx) => {
                            if idx < self.cells.len() {
                                let value = self.cells[idx].clone();
                                self.push(value)?;
                            } else {
                                return Err(VMError::CellError {
                                    index: idx,
                                    message: "Invalid cell index".to_string(),
                                    stack_trace: None,
                                });
                            }
                        }
                        v => {
                            return Err(VMError::TypeError {
                                operation: "cell_get".to_string(),
                                expected: "cell".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                CellSet => {
                    let value = self.pop()?;
                    let cell = self.pop()?;
                    match cell {
                        Value::Cell(idx) => {
                            if idx < self.cells.len() {
                                self.cells[idx] = value;
                                self.push(Value::Nil)?; // CellSet returns nil
                            } else {
                                return Err(VMError::CellError {
                                    index: idx,
                                    message: "Invalid cell index".to_string(),
                                    stack_trace: None,
                                });
                            }
                        }
                        v => {
                            return Err(VMError::TypeError {
                                operation: "cell_set".to_string(),
                                expected: "cell".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                MakeTagged => {
                    let arity = instruction.arg as usize;
                    // Pop all values first
                    let mut values = Vec::with_capacity(arity);
                    for _ in 0..arity {
                        values.push(self.pop()?);
                    }
                    values.reverse();
                    
                    // Then pop the tag
                    let tag_val = self.pop()?;
                    let tag = match tag_val {
                        Value::String(s) => s,
                        v => {
                            return Err(VMError::TypeError {
                                operation: "make_tagged".to_string(),
                                expected: "string".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    };
                    
                    self.push(Value::Tagged { tag, values })?;
                }
                
                GetTag => {
                    let value = self.pop()?;
                    match value {
                        Value::Tagged { tag, .. } => {
                            self.push(Value::String(tag))?;
                        }
                        v => {
                            return Err(VMError::TypeError {
                                operation: "get_tag".to_string(),
                                expected: "tagged value".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                GetTaggedField => {
                    let field_idx = instruction.arg as usize;
                    let value = self.pop()?;
                    match value {
                        Value::Tagged { values, .. } => {
                            if field_idx < values.len() {
                                self.push(values[field_idx].clone())?;
                            } else {
                                return Err(VMError::RuntimeError {
                                    message: format!(
                                        "Tagged field index {} out of bounds (size: {})",
                                        field_idx,
                                        values.len()
                                    ),
                                    stack_trace: None,
                                });
                            }
                        }
                        v => {
                            return Err(VMError::TypeError {
                                operation: "get_tagged_field".to_string(),
                                expected: "tagged value".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                IsTagged => {
                    let expected_tag_idx = instruction.arg as usize;
                    let expected_tag = match self.bytecode.chunks[chunk_id]
                        .constants
                        .get(expected_tag_idx)
                    {
                        Some(Value::String(s)) => s,
                        _ => {
                            return Err(VMError::InvalidConstantIndex {
                                index: expected_tag_idx as u32,
                                max_index: self.bytecode.chunks[chunk_id].constants.len(),
                                stack_trace: None,
                            })
                        }
                    };
                    
                    let value = self.peek(0)?;
                    let is_match = match value {
                        Value::Tagged { tag, .. } => tag == expected_tag,
                        _ => false,
                    };
                    
                    self.push(Value::Boolean(is_match))?;
                }
                
                LoadModule => {
                    let module_name = self.get_constant_string(instruction.arg)?;
                    self.load_module(&module_name)?;
                }
                
                ImportBinding => {
                    // arg encodes module_idx (high 16 bits) and binding_idx (low 16 bits)
                    let module_idx = (instruction.arg >> 16) as usize;
                    let binding_idx = (instruction.arg & 0xFFFF) as usize;
                    
                    let module_name = self.get_constant_string(module_idx as u32)?;
                    let binding_name = self.get_constant_string(binding_idx as u32)?;
                    
                    if let Some(Value::Module { exports, .. }) = self.loaded_modules.get(&module_name) {
                        if let Some(value) = exports.get(&binding_name) {
                            self.push(value.clone())?;
                        } else {
                            return Err(VMError::ModuleError {
                                module_name: module_name.clone(),
                                message: format!("Binding '{}' not found", binding_name),
                                stack_trace: None,
                            });
                        }
                    } else {
                        return Err(VMError::ModuleError {
                            module_name: module_name.clone(),
                            message: "Module not found".to_string(),
                            stack_trace: None,
                        });
                    }
                }
                
                ImportAll => {
                    let module_name = self.get_constant_string(instruction.arg)?;
                    
                    if let Some(Value::Module { exports, .. }) = self.loaded_modules.get(&module_name) {
                        // Import all exports into the current scope
                        for (export_name, value) in exports {
                            // Store each export as a global variable
                            self.globals.insert(export_name.clone(), value.clone());
                        }
                    } else {
                        return Err(VMError::ModuleError {
                            module_name: module_name.clone(),
                            message: "Module not found".to_string(),
                            stack_trace: None,
                        });
                    }
                }
                
                LoadQualified => {
                    // arg encodes module_idx (high 16 bits) and var_idx (low 16 bits)
                    let module_idx = (instruction.arg >> 16) as usize;
                    let var_idx = (instruction.arg & 0xFFFF) as usize;
                    
                    let module_name = self.get_constant_string(module_idx as u32)?;
                    let var_name = self.get_constant_string(var_idx as u32)?;
                    
                    if let Some(Value::Module { exports, .. }) = self.loaded_modules.get(&module_name) {
                        if let Some(value) = exports.get(&var_name) {
                            self.push(value.clone())?;
                        } else {
                            return Err(VMError::ModuleError {
                                module_name: module_name.clone(),
                                message: format!("Export '{}' not found", var_name),
                                stack_trace: None,
                            });
                        }
                    } else {
                        return Err(VMError::ModuleError {
                            module_name: module_name.clone(),
                            message: "Module not found".to_string(),
                            stack_trace: None,
                        });
                    }
                }
                
                BeginModule => {
                    let module_name = self.get_constant_string(instruction.arg)?;
                    self.module_stack
                        .push(self.current_module.clone().unwrap_or_default());
                    self.current_module = Some(module_name);
                }
                
                EndModule => {
                    if let Some(prev_module) = self.module_stack.pop() {
                        self.current_module = if prev_module.is_empty() {
                            None
                        } else {
                            Some(prev_module)
                        };
                    }
                }
                
                ExportBinding => {
                    let binding_name = self.get_constant_string(instruction.arg)?;
                    
                    if let Some(current_module_name) = &self.current_module {
                        let value = self.peek(0)?.clone();
                        
                        // Get or create the module in loaded_modules
                        let module = self
                            .loaded_modules
                            .entry(current_module_name.clone())
                            .or_insert_with(|| Value::Module {
                                name: current_module_name.clone(),
                                exports: FxHashMap::default(),
                            });
                        
                        // Add the export
                        if let Value::Module { exports, .. } = module {
                            exports.insert(binding_name, value);
                        }
                    }
                }
                
                GcAlloc => {
                    let value = self.pop()?;
                    let gc_value = self.gc_alloc(value)?;
                    self.push(gc_value)?;
                }
                
                GcDeref => {
                    let handle = self.pop()?;
                    match handle {
                        Value::GcHandle(any_handle) => {
                            if let Some(gc_handle) = any_handle.downcast_ref::<crate::gc::GcHandle>() {
                                let value = gc_handle.get();
                                self.push(value)?;
                            } else {
                                return Err(VMError::TypeError {
                                    operation: "gc_deref".to_string(),
                                    expected: "GC handle".to_string(),
                                    got: "invalid GC handle type".to_string(),
                                    location: None,
                                    stack_trace: None,
                                });
                            }
                        }
                        v => {
                            return Err(VMError::TypeError {
                                operation: "gc_deref".to_string(),
                                expected: "GC handle".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                GcSet => {
                    let value = self.pop()?;
                    let handle = self.pop()?;
                    match handle {
                        Value::GcHandle(any_handle) => {
                            if let Some(gc_handle) = any_handle.downcast_ref::<crate::gc::GcHandle>() {
                                gc_handle.set(value);
                                self.push(Value::Nil)?;
                            } else {
                                return Err(VMError::TypeError {
                                    operation: "gc_set".to_string(),
                                    expected: "GC handle".to_string(),
                                    got: "invalid GC handle type".to_string(),
                                    location: None,
                                    stack_trace: None,
                                });
                            }
                        }
                        v => {
                            return Err(VMError::TypeError {
                                operation: "gc_set".to_string(),
                                expected: "GC handle".to_string(),
                                got: value_type_name(&v).to_string(),
                                location: None,
                                stack_trace: None,
                            })
                        }
                    }
                }
                
                GcCollect => {
                    self.gc_collect()?;
                    self.push(Value::Nil)?;
                }
                
            Nop => {} // No operation - do nothing
        }
        
        Ok(VMState::Continue)
    }
        
        
            pub fn push(&mut self, value: Value) -> VMResult<()> {
        if self.stack.len() >= STACK_SIZE {
            return Err(VMError::StackOverflow {
                current_depth: self.stack.len(),
                max_depth: STACK_SIZE,
                stack_trace: None,
            });
        }

        // Send debug event
        if self.debug_config.enabled {
            self.debug_config.send_event(VMDebugEvent::StackPush {
                value: value.clone(),
            });
        }

        self.stack.push(value);
        Ok(())
    }

    pub fn pop(&mut self) -> VMResult<Value> {
        let value = self.stack.pop().ok_or_else(|| VMError::StackUnderflow {
            operation: "pop".to_string(),
            stack_size: self.stack.len(),
            stack_trace: None,
        })?;

        // Send debug event
        if self.debug_config.enabled {
            self.debug_config.send_event(VMDebugEvent::StackPop {
                value: value.clone(),
            });
        }

        Ok(value)
    }

    pub fn peek(&self, offset: usize) -> VMResult<&Value> {
        let len = self.stack.len();
        if offset >= len {
            return Err(VMError::StackUnderflow {
                operation: "peek".to_string(),
                stack_size: len,
                stack_trace: None,
            });
        }
        Ok(&self.stack[len - 1 - offset])
    }

    pub fn binary_op<F>(&mut self, op: F) -> VMResult<()>
    where
        F: FnOnce(Value, Value) -> VMResult<Value>,
    {
        let b = self.pop()?;
        let a = self.pop()?;
        let result = op(a, b)?;
        self.push(result)
    }

    pub fn binary_int_op<F>(&mut self, op: F) -> VMResult<()>
    where
        F: FnOnce(i64, i64) -> VMResult<i64>,
    {
        let b = self.pop()?;
        let a = self.pop()?;
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => {
                let result = op(x, y)?;
                self.push(Value::Integer(result))
            }
            (a, b) => Err(VMError::TypeError {
                operation: "binary int operation".to_string(),
                expected: "int".to_string(),
                got: format!("{} and {}", value_type_name(&a), value_type_name(&b)),
                location: None,
                stack_trace: None,
            }),
        }
    }

    fn binary_int_cmp<F>(&mut self, op: F) -> VMResult<()>
    where
        F: FnOnce(i64, i64) -> bool,
    {
        let b = self.pop()?;
        let a = self.pop()?;
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => {
                let result = op(x, y);
                self.push(Value::Boolean(result))
            }
            (a, b) => Err(VMError::TypeError {
                operation: "binary int comparison".to_string(),
                expected: "int".to_string(),
                got: format!("{} and {}", value_type_name(&a), value_type_name(&b)),
                location: None,
                stack_trace: None,
            }),
        }
    }

    pub fn binary_float_op<F>(&mut self, op: F) -> VMResult<()>
    where
        F: FnOnce(f64, f64) -> f64,
    {
        let b = self.pop()?;
        let a = self.pop()?;
        match (a, b) {
            (Value::Float(x), Value::Float(y)) => {
                let result = op(x, y);
                self.push(Value::Float(result))
            }
            (a, b) => Err(VMError::TypeError {
                operation: "binary float operation".to_string(),
                expected: "float".to_string(),
                got: format!("{} and {}", value_type_name(&a), value_type_name(&b)),
                location: None,
                stack_trace: None,
            }),
        }
    }

    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Nil, Value::Nil) => true,
            (Value::Boolean(x), Value::Boolean(y)) => x == y,
            (Value::Integer(x), Value::Integer(y)) => x == y,
            (Value::Float(x), Value::Float(y)) => x == y,
            (Value::String(x), Value::String(y)) => x == y,
            (Value::Symbol(x), Value::Symbol(y)) => x == y,
            (Value::List(x), Value::List(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(a, b)| self.values_equal(a, b))
            }
            (Value::Vector(x), Value::Vector(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(a, b)| self.values_equal(a, b))
            }
            (Value::Map(x), Value::Map(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .all(|(k, v)| y.get(k).map_or(false, |v2| self.values_equal(v, v2)))
            }
            (
                Value::Tagged {
                    tag: tag1,
                    values: vals1,
                },
                Value::Tagged {
                    tag: tag2,
                    values: vals2,
                },
            ) => {
                tag1 == tag2
                    && vals1.len() == vals2.len()
                    && vals1
                        .iter()
                        .zip(vals2)
                        .all(|(a, b)| self.values_equal(a, b))
            }
            (Value::Module { name: n1, .. }, Value::Module { name: n2, .. }) => n1 == n2,
            (Value::Promise(x), Value::Promise(y)) => x == y,
            (Value::Channel(x), Value::Channel(y)) => x == y,
            (Value::Cell(x), Value::Cell(y)) => x == y,
            _ => false,
        }
    }

    pub fn is_truthy(&self, value: &Value) -> bool {
        match value {
            Value::Nil => false,
            Value::Boolean(b) => *b,
            Value::Integer(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::Symbol(_) => true,
            Value::List(l) => !l.is_empty(),
            Value::Procedure(_) => true,
            Value::Vector(v) => !v.is_empty(),
            Value::Map(m) => !m.is_empty(),
            Value::NativeFunction { .. } => true,
            Value::Function { .. } => true,
            Value::Promise(_) => true,
            Value::Channel(_) => true,
            Value::Cell(_) => true,
            Value::Tagged { .. } => true,
            Value::Module { .. } => true,
            Value::GcHandle(_) => true,
            Value::Actor(_) => true,
            Value::Error { .. } => false, // Errors are falsy
            Value::Future { .. } => true,
        }
    }

    pub fn vm_value_to_stdlib_value(&self, value: &Value) -> StdlibValue {
        // Since StdlibValue is just a re-export of Value, we can clone directly
        value.clone()
    }

    pub fn stdlib_value_to_vm_value(&self, value: &StdlibValue) -> Value {
        // Since StdlibValue is just a re-export of Value, we can clone directly
        value.clone()
    }

    fn builtin_to_opcode(&self, name: &str) -> Option<Opcode> {
        match name {
            "+" => Some(Opcode::Add),
            "-" => Some(Opcode::Sub),
            "*" => Some(Opcode::Mul),
            "/" => Some(Opcode::Div),
            "%" => Some(Opcode::Mod),
            "=" | "==" => Some(Opcode::Eq),
            "!=" | "<>" => Some(Opcode::Ne),
            "<" => Some(Opcode::Lt),
            "<=" => Some(Opcode::Le),
            ">" => Some(Opcode::Gt),
            ">=" => Some(Opcode::Ge),
            "and" => Some(Opcode::And),
            "or" => Some(Opcode::Or),
            "not" => Some(Opcode::Not),
            "list-len" | "length" => Some(Opcode::ListLen),
            "list-empty?" | "empty?" => Some(Opcode::ListEmpty),
            "str-len" | "string-length" => Some(Opcode::StrLen),
            "str-concat" | "string-append" => Some(Opcode::StrConcat),
            "str-upper" | "string-upcase" => Some(Opcode::StrUpper),
            "str-lower" | "string-downcase" => Some(Opcode::StrLower),
            _ => None,
        }
    }

    /// Call a value as a function (used by stdlib bridge)
    pub fn call_value(&mut self, arg_count: usize) -> VMResult<()> {
        // Pop arguments
        let mut args = Vec::with_capacity(arg_count);
        for _ in 0..arg_count {
            args.push(self.pop()?);
        }
        args.reverse();

        // Pop function
        let func = self.pop()?;

        match func {
            Value::Function { chunk_id, env } => {
                // Save current env to stack if needed
                let stack_base = self.stack.len();

                // Push arguments back
                for arg in args {
                    self.push(arg)?;
                }

                // Create new call frame
                self.call_stack.push(CallFrame {
                    chunk_id,
                    ip: 0,
                    stack_base,
                    env,
                    start_time: if self.usage_tracker.is_some() {
                        Some(Instant::now())
                    } else {
                        None
                    },
                });

                // Continue execution until this call returns
                let initial_call_depth = self.call_stack.len();
                while self.call_stack.len() >= initial_call_depth {
                    let frame = self
                        .call_stack
                        .last()
                        .ok_or_else(|| VMError::StackUnderflow {
                            operation: "get_current_frame".to_string(),
                            stack_size: self.call_stack.len(),
                            stack_trace: None,
                        })?;
                    let chunk_id = frame.chunk_id;
                    let ip = frame.ip;

                    if ip >= self.bytecode.chunks[chunk_id].instructions.len() {
                        return Err(VMError::InvalidJumpTarget {
                            target: ip,
                            chunk_size: self.bytecode.chunks[chunk_id].instructions.len(),
                            stack_trace: None,
                        });
                    }

                    let instruction = self.bytecode.chunks[chunk_id].instructions[ip].clone();
                    self.call_stack.last_mut().unwrap().ip += 1;

                    match self.execute_instruction(&instruction, chunk_id)? {
                        VMState::Continue => {}
                        VMState::Return => {
                            if self.call_stack.len() == initial_call_depth {
                                // This is our call returning
                                self.call_stack.pop();
                                break;
                            } else {
                                // Inner call returning
                                self.call_stack.pop();
                            }
                        }
                        VMState::Halt => {
                            return Err(VMError::RuntimeError {
                                message: "Unexpected halt in function call".to_string(),
                                stack_trace: Some(self.build_stack_trace()),
                            });
                        }
                    }
                }

                Ok(())
            }
            Value::NativeFunction {
                function, arity, ..
            } => {
                // Check arity
                if args.len() != arity {
                    return Err(VMError::RuntimeError {
                        message: format!(
                            "Native function expects {} arguments, got {}",
                            arity,
                            args.len()
                        ),
                        stack_trace: None,
                    });
                }

                // Call the native function
                let result = function(&args).map_err(|e| VMError::RuntimeError {
                    message: format!("Native function error: {}", e),
                    stack_trace: None,
                })?;

                // Push result
                self.push(result)?;
                Ok(())
            }
            v => Err(VMError::TypeError {
                operation: "call_value".to_string(),
                expected: "function".to_string(),
                got: value_type_name(&v).to_string(),
                location: None,
                stack_trace: None,
            }),
        }
    }

    fn call_handler_function(&mut self, handler: Value, args: Vec<Value>) -> VMResult<Value> {
        // Call a handler function with the given arguments
        match handler {
            Value::Function { chunk_id, env } => {
                // Save current call frame state
                let call_frame = CallFrame {
                    chunk_id,
                    ip: 0,
                    stack_base: self.stack.len(),
                    env,
                    start_time: None,
                };

                // Push arguments onto stack
                for arg in args {
                    self.push(arg)?;
                }

                self.call_stack.push(call_frame);

                // Execute the handler function
                let result;
                let handler_chunk_id = chunk_id; // Save the handler's chunk ID
                loop {
                    // Get current frame
                    let frame = self
                        .call_stack
                        .last()
                        .ok_or_else(|| VMError::RuntimeError {
                            message: "No call frame in handler execution".to_string(),
                            stack_trace: None,
                        })?;

                    // Check if we're still in the handler function
                    if frame.chunk_id != handler_chunk_id {
                        // We've returned from the handler
                        result = self.pop()?;
                        break;
                    }

                    let chunk_id = frame.chunk_id;
                    let ip = frame.ip;

                    // Check bounds
                    let chunk = &self.bytecode.chunks[chunk_id];
                    if ip >= chunk.instructions.len() {
                        return Err(VMError::RuntimeError {
                            message: "Instruction pointer out of bounds in handler".to_string(),
                            stack_trace: Some(self.build_stack_trace()),
                        });
                    }

                    // Get instruction and advance IP
                    let instruction = chunk.instructions[ip].clone();
                    self.call_stack.last_mut().unwrap().ip += 1;

                    match self.execute_instruction(&instruction, chunk_id)? {
                        VMState::Continue => {}
                        VMState::Return => {
                            // The Return opcode already handled everything including
                            // popping the call frame and pushing the return value
                            // We need to pop the return value that was pushed
                            result = self.pop()?;
                            break;
                        }
                        VMState::Halt => {
                            // Lambdas end with Halt, but we treat it like Return
                            // if there's a value on the stack
                            let frame = self.call_stack.last().unwrap();
                            let stack_base = frame.stack_base;

                            if self.stack.len() > stack_base {
                                result = self.pop()?;

                                // Clean up stack to stack_base
                                self.stack.truncate(stack_base);

                                self.call_stack.pop();
                                break;
                            } else {
                                return Err(VMError::RuntimeError {
                                    message: "Handler function halted without return value"
                                        .to_string(),
                                    stack_trace: Some(self.build_stack_trace()),
                                });
                            }
                        }
                    }
                }

                Ok(result)
            }
            Value::NativeFunction {
                function, arity, ..
            } => {
                // Check arity
                if args.len() != arity {
                    return Err(VMError::RuntimeError {
                        message: format!(
                            "Native handler expects {} arguments, got {}",
                            arity,
                            args.len()
                        ),
                        stack_trace: None,
                    });
                }

                // Call the native function directly
                function(&args).map_err(|e| VMError::RuntimeError {
                    message: format!("Native handler error: {}", e),
                    stack_trace: None,
                })
            }
            v => Err(VMError::TypeError {
                operation: "call_handler_function".to_string(),
                expected: "function".to_string(),
                got: value_type_name(&v).to_string(),
                location: None,
                stack_trace: None,
            }),
        }
    }

    fn vm_value_to_core_value(&self, value: &Value) -> fluentai_core::value::Value {
        match value {
            Value::Nil => fluentai_core::value::Value::Nil,
            Value::Boolean(b) => fluentai_core::value::Value::Boolean(*b),
            Value::Integer(i) => fluentai_core::value::Value::Integer(*i),
            Value::Float(f) => fluentai_core::value::Value::Float(*f),
            Value::String(s) => fluentai_core::value::Value::String(s.clone()),
            Value::Symbol(s) => fluentai_core::value::Value::Symbol(s.clone()),
            Value::List(items) => fluentai_core::value::Value::List(
                items
                    .iter()
                    .map(|v| self.vm_value_to_core_value(v))
                    .collect(),
            ),
            Value::Procedure(proc) => fluentai_core::value::Value::Procedure(proc.clone()),
            Value::Vector(items) => fluentai_core::value::Value::Vector(
                items
                    .iter()
                    .map(|v| self.vm_value_to_core_value(v))
                    .collect(),
            ),
            Value::Map(map) => {
                let mut core_map = FxHashMap::default();
                for (k, v) in map.iter() {
                    core_map.insert(k.clone(), self.vm_value_to_core_value(v));
                }
                fluentai_core::value::Value::Map(core_map)
            }
            Value::NativeFunction {
                name,
                arity,
                function,
            } => fluentai_core::value::Value::NativeFunction {
                name: name.clone(),
                arity: *arity,
                function: function.clone(),
            },
            Value::Function { .. } => {
                // Functions can't be directly converted, return a placeholder
                fluentai_core::value::Value::String("<function>".to_string())
            }
            Value::Promise(id) => fluentai_core::value::Value::String(format!("<promise:{}>", id)),
            Value::Channel(id) => fluentai_core::value::Value::String(format!("<channel:{}>", id)),
            Value::Cell(idx) => fluentai_core::value::Value::String(format!("<cell:{}>", idx)),
            Value::Tagged { tag, values } => fluentai_core::value::Value::Tagged {
                tag: tag.clone(),
                values: values
                    .iter()
                    .map(|v| self.vm_value_to_core_value(v))
                    .collect(),
            },
            Value::Module { name, exports } => {
                // Convert to a map representation
                let mut map = FxHashMap::default();
                map.insert(
                    "__module__".to_string(),
                    fluentai_core::value::Value::String(name.clone()),
                );
                let mut export_map = FxHashMap::default();
                for (key, val) in exports {
                    export_map.insert(key.clone(), self.vm_value_to_core_value(val));
                }
                map.insert(
                    "__exports__".to_string(),
                    fluentai_core::value::Value::Map(export_map),
                );
                fluentai_core::value::Value::Map(map)
            }
            Value::GcHandle(_) => fluentai_core::value::Value::String("<gc-handle>".to_string()),
            Value::Actor(id) => fluentai_core::value::Value::String(format!("<actor:{}>", id)),
            Value::Future { .. } => fluentai_core::value::Value::String("<future>".to_string()),
            Value::Error { kind, message, .. } => {
                fluentai_core::value::Value::String(format!("<error:{}:{}>", kind, message))
            }
        }
    }

    fn core_value_to_vm_value(&self, value: &fluentai_core::value::Value) -> Value {
        match value {
            fluentai_core::value::Value::Nil => Value::Nil,
            fluentai_core::value::Value::Boolean(b) => Value::Boolean(*b),
            fluentai_core::value::Value::Integer(i) => Value::Integer(*i),
            fluentai_core::value::Value::Float(f) => Value::Float(*f),
            fluentai_core::value::Value::String(s) => Value::String(s.clone()),
            fluentai_core::value::Value::List(items) => Value::List(
                items
                    .iter()
                    .map(|v| self.core_value_to_vm_value(v))
                    .collect(),
            ),
            fluentai_core::value::Value::Map(map) => {
                let mut vm_map = FxHashMap::default();
                for (k, v) in map.iter() {
                    vm_map.insert(k.clone(), self.core_value_to_vm_value(v));
                }
                Value::Map(vm_map)
            }
            fluentai_core::value::Value::Procedure(_) => {
                // Functions can't be directly converted, return a placeholder
                Value::String("<function>".to_string())
            }
            fluentai_core::value::Value::NativeFunction { name, .. } => {
                // Native functions can't be directly converted, return a placeholder
                Value::String(format!("<native-function: {}>", name))
            }
            fluentai_core::value::Value::Tagged { tag, values } => Value::Tagged {
                tag: tag.clone(),
                values: values
                    .iter()
                    .map(|v| self.core_value_to_vm_value(v))
                    .collect(),
            },
            fluentai_core::value::Value::Symbol(s) => {
                // Preserve symbols as symbols
                Value::Symbol(s.clone())
            }
            fluentai_core::value::Value::Vector(items) => {
                // Convert vectors to lists in VM representation
                Value::List(
                    items
                        .iter()
                        .map(|v| self.core_value_to_vm_value(v))
                        .collect(),
                )
            }
            fluentai_core::value::Value::Function { chunk_id, env } => Value::Function {
                chunk_id: *chunk_id,
                env: env.iter().map(|v| self.core_value_to_vm_value(v)).collect(),
            },
            fluentai_core::value::Value::Promise(id) => Value::Promise(*id),
            fluentai_core::value::Value::Channel(id) => Value::Channel(*id),
            fluentai_core::value::Value::Cell(idx) => Value::Cell(*idx),
            fluentai_core::value::Value::Module { name, exports } => {
                let mut vm_exports = FxHashMap::default();
                for (k, v) in exports {
                    vm_exports.insert(k.clone(), self.core_value_to_vm_value(v));
                }
                Value::Module {
                    name: name.clone(),
                    exports: vm_exports,
                }
            }
            fluentai_core::value::Value::GcHandle(handle) => Value::GcHandle(handle.clone()),
            fluentai_core::value::Value::Actor(id) => Value::Actor(*id),
            fluentai_core::value::Value::Error { kind, message, stack_trace } => Value::Error {
                kind: kind.clone(),
                message: message.clone(),
                stack_trace: stack_trace.clone(),
            },
            fluentai_core::value::Value::Future { chunk_id, env } => Value::Future {
                chunk_id: *chunk_id,
                env: env.iter().map(|v| self.core_value_to_vm_value(v)).collect(),
            },
        }
    }

    // Module system helper methods
    fn get_constant_string(&self, idx: u32) -> VMResult<String> {
        let value = self.bytecode.chunks[self.current_chunk()]
            .constants
            .get(idx as usize)
            .ok_or_else(|| VMError::InvalidConstantIndex {
                index: idx,
                max_index: self.bytecode.chunks[self.current_chunk()].constants.len(),
                stack_trace: None,
            })?;

        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err(VMError::TypeError {
                operation: "get_constant_string".to_string(),
                expected: "string constant".to_string(),
                got: value_type_name(value).to_string(),
                location: None,
                stack_trace: None,
            }),
        }
    }

    fn load_module(&mut self, module_name: &str) -> VMResult<()> {
        // Check if already loaded
        if self.loaded_modules.contains_key(module_name) {
            return Ok(());
        }

        // Load the module file
        let module_info =
            self.module_loader
                .load_module(module_name)
                .map_err(|e| VMError::ModuleError {
                    module_name: module_name.to_string(),
                    message: e.to_string(),
                    stack_trace: None,
                })?;

        // Compile the module
        // Disable optimization to work around optimizer node ID remapping bug
        // TODO: Fix optimizer to properly remap node IDs in Module nodes
        let options = crate::compiler::CompilerOptions {
            optimization_level: fluentai_optimizer::OptimizationLevel::None,
            debug_info: false,
        };
        let compiler = crate::compiler::Compiler::with_options(options);
        let module_bytecode =
            compiler
                .compile(&module_info.graph)
                .map_err(|e| VMError::ModuleError {
                    module_name: module_name.to_string(),
                    message: format!("Failed to compile module: {}", e),
                    stack_trace: None,
                })?;

        // Save current VM state
        let saved_module = self.current_module.clone();
        let saved_stack_len = self.stack.len();
        let saved_globals = self.globals.clone();

        // Set current module context
        self.current_module = Some(module_name.to_string());

        // Create a new VM instance for module execution
        let mut module_vm = VM::new(module_bytecode);
        module_vm.stdlib = self.stdlib.clone();
        module_vm.globals = self.globals.clone();
        module_vm.current_module = Some(module_name.to_string());

        // Execute the module
        let result = module_vm.run();

        // Restore VM state
        self.current_module = saved_module;
        self.stack.truncate(saved_stack_len);

        // Handle execution result
        if let Err(e) = result {
            self.globals = saved_globals;
            return Err(VMError::ModuleError {
                module_name: module_name.to_string(),
                message: format!("Module execution failed: {:?}", e),
                stack_trace: None,
            });
        }

        // Collect exports from module
        let mut exports = FxHashMap::default();
        
        // First check if the module registered its exports via ExportBinding
        if let Some(Value::Module { exports: module_exports, .. }) = module_vm.loaded_modules.get(module_name) {
            // Use the exports from ExportBinding
            exports = module_exports.clone();
        } else {
            // Fall back to collecting from globals (for modules using define)
            for export_name in &module_info.exports {
                if let Some(value) = module_vm.globals.get(export_name) {
                    exports.insert(export_name.clone(), value.clone());
                }
            }
        }

        // Create module value with actual exports
        let module_value = Value::Module {
            name: module_name.to_string(),
            exports,
        };

        self.loaded_modules
            .insert(module_name.to_string(), module_value);

        Ok(())
    }

    fn current_chunk(&self) -> usize {
        self.call_stack
            .last()
            .map(|frame| frame.chunk_id)
            .unwrap_or(self.bytecode.main_chunk)
    }

    /// Set a global variable
    pub fn set_global(&mut self, name: String, value: Value) {
        self.globals.insert(name, value);
    }

    pub fn get_global(&self, name: &str) -> Option<&Value> {
        self.globals.get(name)
    }

    /// Build a stack trace from current call stack
    pub fn build_stack_trace(&self) -> StackTrace {
        let mut trace = StackTrace::new();

        for frame in &self.call_stack {
            let function_name = self
                .bytecode
                .chunks
                .get(frame.chunk_id)
                .and_then(|chunk| chunk.name.clone())
                .unwrap_or_else(|| format!("<anonymous:{}>", frame.chunk_id));

            // Get source location from source map if available
            let location = self
                .bytecode
                .chunks
                .get(frame.chunk_id)
                .and_then(|chunk| chunk.source_map.as_ref())
                .and_then(|source_map| source_map.get_location(frame.ip))
                .map(|src_loc| crate::error::SourceLocation {
                    file: self.bytecode.chunks.get(frame.chunk_id)
                        .and_then(|chunk| chunk.source_map.as_ref())
                        .and_then(|sm| sm.filename.clone()),
                    line: src_loc.line.unwrap_or(0) as usize,
                    column: src_loc.column.unwrap_or(0) as usize,
                    function: Some(function_name.clone()),
                });

            trace.push_frame(StackFrame {
                function_name,
                chunk_id: frame.chunk_id,
                ip: frame.ip,
                location,
            });
        }

        trace
    }
    
    /// Create an error with source location from current instruction
    pub fn create_error_with_location(&self, mut error: VMError) -> VMError {
        // Add stack trace if not already present
        match &mut error {
            VMError::StackOverflow { stack_trace, .. } |
            VMError::StackUnderflow { stack_trace, .. } |
            VMError::CallStackOverflow { stack_trace, .. } |
            VMError::TypeError { stack_trace, .. } |
            VMError::DivisionByZero { stack_trace, .. } |
            VMError::IntegerOverflow { stack_trace, .. } |
            VMError::InvalidConstantIndex { stack_trace, .. } |
            VMError::InvalidLocalIndex { stack_trace, .. } |
            VMError::InvalidJumpTarget { stack_trace, .. } |
            VMError::ResourceLimitExceeded { stack_trace, .. } |
            VMError::ModuleError { stack_trace, .. } |
            VMError::AsyncError { stack_trace, .. } |
            VMError::CellError { stack_trace, .. } |
            VMError::UnknownIdentifier { stack_trace, .. } |
            VMError::RuntimeError { stack_trace, .. } => {
                if stack_trace.is_none() {
                    *stack_trace = Some(self.build_stack_trace());
                }
            }
            _ => {}
        }
        
        // Add source location if not already present and applicable
        if let Some(frame) = self.call_stack.last() {
            match &mut error {
                VMError::TypeError { location, .. } |
                VMError::DivisionByZero { location, .. } |
                VMError::InvalidOpcode { location, .. } |
                VMError::UnknownIdentifier { location, .. } => {
                    if location.is_none() {
                        *location = self
                            .bytecode
                            .chunks
                            .get(frame.chunk_id)
                            .and_then(|chunk| chunk.source_map.as_ref())
                            .and_then(|source_map| source_map.get_location(frame.ip))
                            .map(|src_loc| crate::error::SourceLocation {
                                file: self.bytecode.chunks.get(frame.chunk_id)
                                    .and_then(|chunk| chunk.source_map.as_ref())
                                    .and_then(|sm| sm.filename.clone()),
                                line: src_loc.line.unwrap_or(0) as usize,
                                column: src_loc.column.unwrap_or(0) as usize,
                                function: self.bytecode.chunks.get(frame.chunk_id)
                                    .and_then(|chunk| chunk.name.clone()),
                            });
                    }
                }
                _ => {}
            }
        }
        
        error
    }

    /// Set resource limits
    pub fn set_resource_limits(&mut self, limits: ResourceLimits) {
        self.resource_limits = limits;
    }
    
    // ===== Public accessor methods for opcode handlers =====
    
    // Stack operations
    pub fn stack_len(&self) -> usize {
        self.stack.len()
    }
    
    pub fn stack_swap(&mut self, a: usize, b: usize) {
        self.stack.swap(a, b);
    }
    
    // Constant access
    pub fn get_constant(&self, chunk_id: usize, index: usize) -> VMResult<&Value> {
        self.bytecode.chunks[chunk_id]
            .constants
            .get(index)
            .ok_or_else(|| VMError::InvalidConstantIndex {
                index: index as u32,
                max_index: self.bytecode.chunks[chunk_id].constants.len(),
                stack_trace: None,
            })
    }
    
    pub fn get_constant_string_at(&self, chunk_id: usize, index: usize) -> VMResult<String> {
        match self.get_constant(chunk_id, index)? {
            Value::String(s) => Ok(s.clone()),
            _ => Err(VMError::InvalidConstantIndex {
                index: index as u32,
                max_index: self.bytecode.chunks[chunk_id].constants.len(),
                stack_trace: None,
            }),
        }
    }
    
    // Control flow
    pub fn set_ip(&mut self, ip: usize) {
        if let Some(frame) = self.call_stack.last_mut() {
            frame.ip = ip;
        }
    }
    
    
    pub fn call_stack_len(&self) -> usize {
        self.call_stack.len()
    }
    
    pub fn push_call_frame(&mut self, frame: CallFrame) -> VMResult<()> {
        if self.call_stack.len() >= self.resource_limits.max_call_depth {
            return Err(VMError::CallStackOverflow {
                current_depth: self.call_stack.len(),
                max_depth: self.resource_limits.max_call_depth,
                stack_trace: None,
            });
        }
        self.call_stack.push(frame);
        Ok(())
    }
    
    pub fn pop_call_frame_with_return(&mut self, return_val: Value) -> VMResult<()> {
        if let Some(frame) = self.call_stack.pop() {
            // Track execution time if usage tracking is enabled
            if let Some(tracker) = &self.usage_tracker {
                if let Some(start_time) = frame.start_time {
                    let elapsed = start_time.elapsed().as_nanos() as u64;
                    if let Ok(mut tracker_guard) = tracker.write() {
                        tracker_guard.record_execution(frame.chunk_id, elapsed);
                    }
                }
            }
            
            // Restore stack to frame base and push return value
            self.stack.truncate(frame.stack_base);
            self.push(return_val)?;
        }
        Ok(())
    }
    
    pub fn has_usage_tracker(&self) -> bool {
        self.usage_tracker.is_some()
    }
    
    /// Check if a chunk should be JIT compiled
    #[cfg(feature = "jit")]
    pub fn should_jit_compile(&self, chunk_id: usize) -> bool {
        if let Some(tracker) = &self.usage_tracker {
            if let Ok(tracker_guard) = tracker.read() {
                if let Some(stats) = tracker_guard.get_stats_for_chunk(chunk_id) {
                    return self.jit_manager.should_compile(&stats);
                }
            }
        }
        false
    }
    
    #[cfg(not(feature = "jit"))]
    pub fn should_jit_compile(&self, _chunk_id: usize) -> bool {
        false
    }
    
    /// Try to execute a function using JIT compilation
    #[cfg(feature = "jit")]
    pub fn try_jit_execute(&mut self, chunk_id: usize) -> VMResult<Option<Value>> {
        // First attempt compilation if needed
        if let Some(tracker) = &self.usage_tracker {
            if let Ok(tracker_guard) = tracker.read() {
                if let Some(stats) = tracker_guard.get_stats_for_chunk(chunk_id) {
                    if self.jit_manager.should_compile(&stats) {
                        drop(tracker_guard);
                        self.jit_manager.compile_chunk(chunk_id, &self.bytecode)?;
                    }
                }
            }
        }
        
        // Try to execute the JIT-compiled version
        match self.jit_manager.execute_if_compiled(chunk_id, &self.bytecode) {
            Some(Ok(value)) => Ok(Some(value)),
            Some(Err(_)) => Ok(None), // Fall back to interpreter
            None => Ok(None),
        }
    }
    
    #[cfg(not(feature = "jit"))]
    pub fn try_jit_execute(&mut self, _chunk_id: usize) -> VMResult<Option<Value>> {
        Ok(None)
    }
    
    /// Get JIT compilation statistics
    #[cfg(feature = "jit")]
    pub fn jit_stats(&self) -> &crate::jit_integration::JitStats {
        self.jit_manager.stats()
    }
    
    pub fn emit_function_call_debug_event(&self, func: &Value, arg_count: usize) {
        if self.debug_config.enabled {
            self.debug_config.send_event(VMDebugEvent::FunctionCall {
                name: match func {
                    Value::Function { .. } => Some("lambda".to_string()),
                    Value::NativeFunction { name, .. } => Some(name.clone()),
                    _ => None,
                },
                arg_count,
                call_depth: self.call_stack.len(),
            });
        }
    }
    
    pub fn value_type_name(&self, value: &Value) -> &str {
        value_type_name(value)
    }
    
    // ===== Async support methods =====
    
    pub fn create_channel(&mut self) -> ChannelId {
        let channel_id = self.id_generator.next_channel_id();
        let (tx, rx) = mpsc::channel(100); // Default capacity
        self.channels.insert(channel_id, (tx, rx));
        channel_id
    }
    
    pub fn send_to_channel(&mut self, channel_id: ChannelId, value: Value) -> VMResult<()> {
        if let Some((tx, _)) = self.channels.get(&channel_id) {
            tx.try_send(value).map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => VMError::AsyncError {
                    message: "Channel buffer full".to_string(),
                    stack_trace: None,
                },
                mpsc::error::TrySendError::Closed(_) => VMError::AsyncError {
                    message: "Channel closed".to_string(),
                    stack_trace: None,
                },
            })
        } else {
            Err(VMError::UnknownIdentifier {
                name: format!("channel:{}", channel_id.0),
                location: None,
                stack_trace: None,
            })
        }
    }
    
    pub fn receive_from_channel(&mut self, channel_id: ChannelId) -> VMResult<Value> {
        if let Some((_, rx)) = self.channels.get_mut(&channel_id) {
            match rx.try_recv() {
                Ok(value) => Ok(value),
                Err(mpsc::error::TryRecvError::Empty) => {
                    // For synchronous execution, return Nil on empty channel
                    Ok(Value::Nil)
                },
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    Err(VMError::AsyncError {
                        message: "Channel disconnected".to_string(),
                        stack_trace: None,
                    })
                }
            }
        } else {
            Err(VMError::UnknownIdentifier {
                name: format!("channel:{}", channel_id.0),
                location: None,
                stack_trace: None,
            })
        }
    }
    
    pub fn get_channel_receiver_mut(&mut self, channel_id: &ChannelId) -> Option<&mut mpsc::Receiver<Value>> {
        self.channels.get_mut(channel_id).map(|(_, rx)| rx)
    }
    
    pub fn await_promise(&mut self, promise_id: PromiseId) -> VMResult<Value> {
        if let Some(mut receiver) = self.promises.remove(&promise_id) {
            // Blocking receive for synchronous execution
            match receiver.try_recv() {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(e)) => Err(e),
                Err(_) => {
                    // Channel not ready, return a placeholder
                    Ok(Value::Nil)
                }
            }
        } else {
            Err(VMError::AsyncError {
                message: format!("Promise {:?} not found", promise_id),
                stack_trace: None,
            })
        }
    }
    
    pub fn spawn_task(&mut self, func: Value) -> VMResult<()> {
        match func {
            Value::Function { chunk_id, env } => {
                // Create a promise for the result
                let promise_id = self.id_generator.next_promise_id();
                let (tx, rx) = oneshot::channel();
                
                // Store the receiver
                self.promises.insert(promise_id, rx);
                
                // Clone shared resources
                let bytecode = Arc::clone(&self.bytecode);
                let stdlib = self.stdlib.clone();
                let effect_runtime = Arc::clone(&self.effect_runtime);
                let effect_context = Arc::clone(&self.effect_context);
                let globals = self.globals.clone(); // COW clone
                
                // Spawn the task
                tokio::spawn(async move {
                    // Create a new VM with shared bytecode
                    let mut task_vm = VM::with_shared_bytecode(bytecode);
                    task_vm.stdlib = stdlib;
                    task_vm.effect_runtime = effect_runtime;
                    task_vm.effect_context = effect_context;
                    task_vm.globals = globals;
                    
                    // Set up the call frame for the function
                    task_vm.call_stack.push(CallFrame {
                        chunk_id,
                        ip: 0,
                        stack_base: 0,
                        env,
                        start_time: None,
                    });
                    
                    // Run the function
                    let result = task_vm.run_inner();
                    
                    // Send the result through the promise channel
                    let _ = tx.send(result);
                });
                
                // Push the promise ID
                self.push(Value::Promise(promise_id.0))?;
                Ok(())
            }
            _ => Err(VMError::TypeError {
                operation: "spawn".to_string(),
                expected: "function".to_string(),
                got: value_type_name(&func).to_string(),
                location: None,
                stack_trace: None,
            }),
        }
    }
    
    pub fn create_actor(&mut self, initial_state: Value, handler: Value) -> VMResult<ActorId> {
        // Validate handler is a function
        match &handler {
            Value::Function { .. } | Value::Procedure(_) => {},
            _ => {
                return Err(VMError::TypeError {
                    operation: "create_actor".to_string(),
                    expected: "function".to_string(),
                    got: value_type_name(&handler).to_string(),
                    location: None,
                    stack_trace: None,
                })
            }
        }
        
        // Create actor ID and mailbox
        let actor_id = self.id_generator.next_actor_id();
        let (tx, rx) = mpsc::channel(100); // Buffer size of 100
        
        // Create actor
        let actor = Actor {
            state: initial_state,
            handler,
            mailbox: rx,
            sender: tx,
        };
        
        // Store actor
        self.actors.insert(actor_id, actor);
        
        Ok(actor_id)
    }
    
    pub fn send_to_actor(&mut self, actor_id: ActorId, message: Value) -> VMResult<()> {
        if let Some(actor) = self.actors.get(&actor_id) {
            // Send message (non-blocking)
            actor.sender.try_send(message).map_err(|_| VMError::AsyncError {
                message: "Actor mailbox full or closed".to_string(),
                stack_trace: None,
            })
        } else {
            Err(VMError::UnknownIdentifier {
                name: format!("actor:{}", actor_id.0),
                location: None,
                stack_trace: None,
            })
        }
    }
    
    /// Process messages for a specific actor
    pub fn process_actor_messages(&mut self, actor_id: ActorId) -> VMResult<()> {
        // Get the actor (we need to temporarily remove it to avoid borrow issues)
        let mut actor = self.actors.remove(&actor_id).ok_or_else(|| VMError::UnknownIdentifier {
            name: format!("actor:{}", actor_id.0),
            location: None,
            stack_trace: None,
        })?;
        
        // Set current actor context
        self.current_actor = Some(actor_id);
        
        // Try to receive a message from the mailbox
        while let Ok(message) = actor.mailbox.try_recv() {
            // Set the current message for ActorReceive opcode
            self.current_actor_message = Some(message.clone());
            
            // Call the handler function with (state, message)
            let handler = actor.handler.clone();
            
            // Push handler, state, and message onto stack
            self.push(handler)?;
            self.push(actor.state.clone())?;
            self.push(message)?;
            
            // Call the handler with 2 arguments
            self.call_value(2)?;
            
            // The result is the new state
            actor.state = self.pop()?;
            
            // Clear the current message
            self.current_actor_message = None;
        }
        
        // Clear actor context
        self.current_actor = None;
        
        // Put the actor back
        self.actors.insert(actor_id, actor);
        Ok(())
    }
    
    /// Process messages for all actors (typically called periodically)
    pub fn process_all_actor_messages(&mut self) -> VMResult<()> {
        let actor_ids: Vec<ActorId> = self.actors.keys().cloned().collect();
        for actor_id in actor_ids {
            self.process_actor_messages(actor_id)?;
        }
        Ok(())
    }
    
    /// Get current actor context (if any)
    pub fn current_actor_context(&self) -> Option<ActorId> {
        self.current_actor
    }
    
    /// Update the state of an actor (used by Become)
    pub fn update_actor_state(&mut self, actor_id: ActorId, new_state: Value) -> VMResult<()> {
        if let Some(actor) = self.actors.get_mut(&actor_id) {
            actor.state = new_state;
            Ok(())
        } else {
            Err(VMError::UnknownIdentifier {
                name: format!("actor:{}", actor_id.0),
                location: None,
                stack_trace: None,
            })
        }
    }
    
    /// Get the current actor message (for ActorReceive opcode)
    pub fn get_current_actor_message(&self) -> Option<Value> {
        self.current_actor_message.clone()
    }
    
    pub fn take_promise(&mut self, promise_id: &PromiseId) -> Option<oneshot::Receiver<VMResult<Value>>> {
        self.promises.remove(promise_id)
    }
    
    // Memory operations
    pub fn get_local(&self, index: usize) -> VMResult<&Value> {
        let frame = self.call_stack.last()
            .ok_or_else(|| VMError::RuntimeError {
                message: "No active call frame".to_string(),
                stack_trace: None,
            })?;
        
        let stack_idx = frame.stack_base + index;
        self.stack.get(stack_idx)
            .ok_or_else(|| VMError::RuntimeError {
                message: format!("Invalid local variable index: {}", index),
                stack_trace: None,
            })
    }
    
    pub fn set_local(&mut self, index: usize, value: Value) -> VMResult<()> {
        let frame = self.call_stack.last()
            .ok_or_else(|| VMError::RuntimeError {
                message: "No active call frame".to_string(),
                stack_trace: None,
            })?;
        
        let stack_idx = frame.stack_base + index;
        if stack_idx < self.stack.len() {
            self.stack[stack_idx] = value;
            Ok(())
        } else {
            Err(VMError::RuntimeError {
                message: format!("Invalid local variable index: {}", index),
                stack_trace: None,
            })
        }
    }
    
    
    pub fn define_global(&mut self, name: String, value: Value) -> VMResult<()> {
        // In this implementation, define is the same as set
        self.globals.insert(name, value);
        Ok(())
    }
    
    pub fn is_stdlib_function(&self, name: &str) -> bool {
        self.stdlib.contains(name)
    }
    
    pub fn is_builtin(&self, name: &str) -> bool {
        self.builtin_to_opcode(name).is_some()
    }
    
    // Upvalue operations
    pub fn get_upvalue(&self, index: usize) -> VMResult<&Value> {
        let frame = self.call_stack.last()
            .ok_or_else(|| VMError::RuntimeError {
                message: "No active call frame".to_string(),
                stack_trace: None,
            })?;
        
        frame.env.get(index)
            .ok_or_else(|| VMError::RuntimeError {
                message: format!("Invalid upvalue index: {}", index),
                stack_trace: None,
            })
    }
    
    pub fn set_upvalue(&mut self, index: usize, value: Value) -> VMResult<()> {
        let frame = self.call_stack.last_mut()
            .ok_or_else(|| VMError::RuntimeError {
                message: "No active call frame".to_string(),
                stack_trace: None,
            })?;
        
        if index < frame.env.len() {
            frame.env[index] = value;
            Ok(())
        } else {
            Err(VMError::RuntimeError {
                message: format!("Invalid upvalue index: {}", index),
                stack_trace: None,
            })
        }
    }
    
    // Cell operations
    pub fn create_cell(&mut self, value: Value) -> usize {
        let id = self.cells.len();
        self.cells.push(value);
        id
    }
    
    pub fn get_cell_value(&self, id: usize) -> VMResult<&Value> {
        self.cells.get(id)
            .ok_or_else(|| VMError::RuntimeError {
                message: format!("Invalid cell id: {}", id),
                stack_trace: None,
            })
    }
    
    pub fn set_cell_value(&mut self, id: usize, value: Value) -> VMResult<()> {
        if id < self.cells.len() {
            self.cells[id] = value;
            Ok(())
        } else {
            Err(VMError::RuntimeError {
                message: format!("Invalid cell id: {}", id),
                stack_trace: None,
            })
        }
    }
    
    // Native function calls
    pub fn call_native_function(&mut self, native_func: &str, args: Vec<Value>) -> VMResult<()> {
        // Look up the native function in globals
        if let Some(func_value) = self.globals.get(native_func).cloned() {
            if let Value::NativeFunction { function, .. } = func_value {
                // Call the native function
                match function(&args) {
                    Ok(result) => {
                        self.push(result)?;
                        Ok(())
                    }
                    Err(e) => Err(VMError::RuntimeError {
                        message: format!("Native function error: {}", e),
                        stack_trace: None,
                    }),
                }
            } else {
                Err(VMError::TypeError {
                    operation: "call".to_string(),
                    expected: "native function".to_string(),
                    got: "non-function value".to_string(),
                    location: None,
                    stack_trace: None,
                })
            }
        } else {
            Err(VMError::UnknownIdentifier {
                name: native_func.to_string(),
                location: None,
                stack_trace: None,
            })
        }
    }
    
    // Tail call support
    pub fn setup_tail_call(&mut self, func: Value, args: Vec<Value>) -> VMResult<()> {
        // Implementation would reuse current call frame
        Ok(())
    }
    
    pub fn handle_tail_return(&mut self, result: Value) -> VMResult<()> {
        if let Some(frame) = self.call_stack.pop() {
            self.stack.truncate(frame.stack_base);
            self.push(result)?;
        }
        Ok(())
    }
    
    
    // Effect operations
    pub fn perform_effect(&mut self, operation: String) -> VMResult<()> {
        // Simplified implementation
        Ok(())
    }
    
    pub fn install_effect_handlers(&mut self, handlers: Vec<Value>) -> VMResult<()> {
        // Simplified implementation
        Ok(())
    }
    
    pub fn uninstall_effect_handler(&mut self) -> VMResult<()> {
        // Simplified implementation
        Ok(())
    }
    
    pub fn resume_from_handler(&mut self, value: Value) -> VMResult<()> {
        // Simplified implementation
        Ok(())
    }
    
    // Error handling
    pub fn push_error_handler(&mut self, catch_ip: usize, finally_ip: Option<usize>) -> VMResult<()> {
        // Simplified implementation
        Ok(())
    }
    
    pub fn pop_error_handler(&mut self) -> VMResult<()> {
        // Simplified implementation
        Ok(())
    }
    
    pub fn throw_error(&mut self, error: Value) -> VMResult<VMState> {
        // Simplified implementation
        Ok(VMState::Continue)
    }
    
    pub fn start_finally_block(&mut self) -> VMResult<()> {
        // Simplified implementation
        Ok(())
    }
    
    pub fn end_finally_block(&mut self) -> VMResult<()> {
        // Simplified implementation
        Ok(())
    }
    
    // ===== Async support accessor methods =====
    
    /// Get a reference to the bytecode
    pub fn bytecode(&self) -> &Bytecode {
        &self.bytecode
    }
    
    /// Get a reference to the stack
    pub fn stack(&self) -> &[Value] {
        &self.stack
    }
    
    /// Get a mutable reference to the stack
    pub fn stack_mut(&mut self) -> &mut Vec<Value> {
        &mut self.stack
    }
    
    /// Get a reference to the call stack
    pub fn call_stack(&self) -> &[CallFrame] {
        &self.call_stack
    }
    
    /// Get a mutable reference to the call stack
    pub fn call_stack_mut(&mut self) -> &mut Vec<CallFrame> {
        &mut self.call_stack
    }
    
    
    /// Get a reference to a channel
    pub fn get_channel(&self, channel_id: &crate::safety::ChannelId) -> Option<&(mpsc::Sender<Value>, mpsc::Receiver<Value>)> {
        self.channels.get(channel_id)
    }
    
    /// Get a mutable reference to a channel
    pub fn get_channel_mut(&mut self, channel_id: &crate::safety::ChannelId) -> Option<&mut (mpsc::Sender<Value>, mpsc::Receiver<Value>)> {
        self.channels.get_mut(channel_id)
    }
    
}

#[derive(Debug)]
pub enum VMState {
    Continue,
    Return,
    Halt,
}

#[cfg(test)]
mod inline_tests {
    use super::*;
    use fluentai_bytecode::{Bytecode, BytecodeChunk, Instruction, Opcode};

    #[test]
    fn test_vm_creation_inline() {
        let mut bytecode = Bytecode::new();
        let mut chunk = BytecodeChunk::new(Some("test".to_string()));
        chunk.add_instruction(Instruction::new(Opcode::Halt));
        bytecode.chunks.push(chunk);

        let vm = VM::new(bytecode);
        assert_eq!(vm.stack.len(), 0);
        assert_eq!(vm.globals.len(), 0);
        assert_eq!(vm.call_stack.len(), 0);
    }

    #[test]
    fn test_usage_tracker_inline() {
        let mut tracker = UsageTracker::new();
        let node_id = NodeId(std::num::NonZeroU32::new(1).unwrap());

        tracker.register_chunk(0, node_id);
        tracker.record_execution(0, 1000);

        let stats = tracker.get_stats(node_id);
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().execution_count, 1);
    }

    #[test]
    fn test_call_frame_creation() {
        let frame = CallFrame {
            chunk_id: 0,
            ip: 0,
            stack_base: 0,
            env: vec![],
            start_time: Some(Instant::now()),
        };

        assert_eq!(frame.chunk_id, 0);
        assert_eq!(frame.ip, 0);
        assert_eq!(frame.stack_base, 0);
        assert!(frame.env.is_empty());
        assert!(frame.start_time.is_some());
    }

    #[test]
    fn test_usage_tracker_error_recording() {
        let mut tracker = UsageTracker::new();
        let node_id = NodeId(std::num::NonZeroU32::new(1).unwrap());

        tracker.register_chunk(0, node_id);
        tracker.record_error(0);

        let stats = tracker.get_stats(node_id);
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().error_count, 1);
    }

    #[test]
    fn test_usage_tracker_hot_path() {
        let mut tracker = UsageTracker::new();
        let node_id = NodeId(std::num::NonZeroU32::new(1).unwrap());

        tracker.register_chunk(0, node_id);

        // Execute many times to trigger hot path detection
        for _ in 0..1001 {
            tracker.record_execution(0, 100);
        }

        let stats = tracker.get_stats(node_id);
        assert!(stats.is_some());
        assert!(stats.unwrap().is_hot_path);
    }

    #[test]
    fn test_usage_tracker_all_stats() {
        let mut tracker = UsageTracker::new();
        let node_id1 = NodeId(std::num::NonZeroU32::new(1).unwrap());
        let node_id2 = NodeId(std::num::NonZeroU32::new(2).unwrap());

        tracker.register_chunk(0, node_id1);
        tracker.register_chunk(1, node_id2);

        tracker.record_execution(0, 100);
        tracker.record_execution(1, 200);

        let all_stats = tracker.get_all_stats();
        assert_eq!(all_stats.len(), 2);
        assert!(all_stats.contains_key(&node_id1));
        assert!(all_stats.contains_key(&node_id2));
    }

    #[test]
    fn test_value_conversions() {
        let mut bytecode = Bytecode::new();
        let mut chunk = BytecodeChunk::new(Some("test".to_string()));
        chunk.add_instruction(Instruction::new(Opcode::Halt));
        bytecode.chunks.push(chunk);

        let vm = VM::new(bytecode);

        // Test VM value to stdlib value conversion
        let vm_val = Value::Integer(42);
        let stdlib_val = vm.vm_value_to_stdlib_value(&vm_val);
        match stdlib_val {
            StdlibValue::Integer(i) => assert_eq!(i, 42),
            _ => panic!("Expected integer"),
        }

        // Test list conversion
        let vm_list = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        let stdlib_list = vm.vm_value_to_stdlib_value(&vm_list);
        match stdlib_list {
            StdlibValue::List(_) => {}
            _ => panic!("Expected list"),
        }
    }
}