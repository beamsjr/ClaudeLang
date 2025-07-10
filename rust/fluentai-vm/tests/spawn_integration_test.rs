use fluentai_core::ast::{Graph, Literal, Node};
use fluentai_core::value::Value;
use fluentai_effects::{EffectContext, EffectRuntime, handlers::*};
use fluentai_vm::{compiler::Compiler, VM};
use std::sync::Arc;

#[test]
fn test_spawn_with_lambda() {
    // Initialize effect context and runtime
    let context = Arc::new(EffectContext::new());
    // Create a new runtime since we're not in an async context
    let runtime = Arc::new(EffectRuntime::new().unwrap());
    
    // Register handlers
    context.register_handler(Arc::new(IOHandler::new()));
    context.register_handler(Arc::new(AsyncHandler::new()));
    context.register_handler(Arc::new(ConcurrentHandler::new()));

    // Create a graph for: (let ((promise (spawn (lambda () (+ 1 2))))) (await promise))
    let mut graph = Graph::new();

    // Create lambda: (lambda () (+ 1 2))
    let plus = graph.add_node(Node::Variable { name: "+".to_string() }).unwrap();
    let one = graph.add_node(Node::Literal(Literal::Integer(1))).unwrap();
    let two = graph.add_node(Node::Literal(Literal::Integer(2))).unwrap();
    let add = graph.add_node(Node::Application {
        function: plus,
        args: vec![one, two],
    }).unwrap();
    let lambda = graph.add_node(Node::Lambda {
        params: vec![],
        body: add,
    }).unwrap();

    // Create spawn: (spawn lambda)
    let spawn = graph.add_node(Node::Spawn { expr: lambda }).unwrap();

    // Create await: (await promise)
    let promise_var = graph.add_node(Node::Variable { name: "promise".to_string() }).unwrap();
    let await_node = graph.add_node(Node::Await { expr: promise_var }).unwrap();

    // Create let binding: (let ((promise spawn)) await)
    let let_node = graph.add_node(Node::Let {
        bindings: vec![("promise".to_string(), spawn)],
        body: await_node,
    }).unwrap();

    graph.root_id = Some(let_node);

    // Compile
    let compiler = Compiler::new();
    let bytecode = compiler.compile(&graph).unwrap();

    // Create VM with effect context and runtime
    let mut vm = VM::new(bytecode);
    vm.set_effect_context(context);
    vm.set_effect_runtime(runtime);

    // Run the VM (spawned tasks will execute in the background)
    let result = vm.run().expect("VM execution failed");

    // Verify result
    assert_eq!(result, Value::Integer(3))
}

#[test]
fn test_spawn_with_channel() {
    // Initialize effect context and runtime
    let context = Arc::new(EffectContext::new());
    // Create a new runtime since we're not in an async context
    let runtime = Arc::new(EffectRuntime::new().unwrap());
    
    // Register handlers
    context.register_handler(Arc::new(IOHandler::new()));
    context.register_handler(Arc::new(AsyncHandler::new()));
    context.register_handler(Arc::new(ConcurrentHandler::new()));

    // Create a graph for: (let ((ch (chan 1))) (spawn (lambda () (send! ch 42))) (receive! ch))
    let mut graph = Graph::new();

    // Create channel: (chan 1)
    let channel = graph.add_node(Node::Channel { capacity: None }).unwrap();

    // Create lambda: (lambda () (send! ch 42))
    let ch_var = graph.add_node(Node::Variable { name: "ch".to_string() }).unwrap();
    let value = graph.add_node(Node::Literal(Literal::Integer(42))).unwrap();
    let send = graph.add_node(Node::Send { channel: ch_var, value }).unwrap();
    let lambda = graph.add_node(Node::Lambda {
        params: vec![],
        body: send,
    }).unwrap();

    // Create spawn: (spawn lambda)
    let spawn = graph.add_node(Node::Spawn { expr: lambda }).unwrap();

    // Create receive: (receive! ch)
    let ch_var2 = graph.add_node(Node::Variable { name: "ch".to_string() }).unwrap();
    let receive = graph.add_node(Node::Receive { channel: ch_var2 }).unwrap();

    // Create sequence: spawn then receive
    let sequence = graph.add_node(Node::List(vec![spawn, receive])).unwrap();

    // Create let binding: (let ((ch channel)) sequence)
    let let_node = graph.add_node(Node::Let {
        bindings: vec![("ch".to_string(), channel)],
        body: sequence,
    }).unwrap();

    graph.root_id = Some(let_node);

    // Compile
    let compiler = Compiler::new();
    let bytecode = compiler.compile(&graph).unwrap();

    // Create VM with effect context and runtime
    let mut vm = VM::new(bytecode);
    vm.set_effect_context(context);
    vm.set_effect_runtime(runtime);

    // Run the VM (spawned tasks will execute in the background)
    let result = vm.run().expect("VM execution failed");

    // Verify result (should be the last expression, which is the receive result)
    assert_eq!(result, Value::Integer(42))
}

#[test]
fn test_multiple_spawns() {
    // Initialize effect context and runtime
    let context = Arc::new(EffectContext::new());
    // Create a new runtime since we're not in an async context
    let runtime = Arc::new(EffectRuntime::new().unwrap());
    
    // Register handlers
    context.register_handler(Arc::new(IOHandler::new()));
    context.register_handler(Arc::new(AsyncHandler::new()));
    context.register_handler(Arc::new(ConcurrentHandler::new()));

    // Create a graph for: (let ((p1 (spawn (lambda () 1))) (p2 (spawn (lambda () 2)))) (list (await p1) (await p2)))
    let mut graph = Graph::new();

    // Create first lambda: (lambda () 1)
    let one = graph.add_node(Node::Literal(Literal::Integer(1))).unwrap();
    let lambda1 = graph.add_node(Node::Lambda {
        params: vec![],
        body: one,
    }).unwrap();
    let spawn1 = graph.add_node(Node::Spawn { expr: lambda1 }).unwrap();

    // Create second lambda: (lambda () 2)
    let two = graph.add_node(Node::Literal(Literal::Integer(2))).unwrap();
    let lambda2 = graph.add_node(Node::Lambda {
        params: vec![],
        body: two,
    }).unwrap();
    let spawn2 = graph.add_node(Node::Spawn { expr: lambda2 }).unwrap();

    // Create awaits
    let p1_var = graph.add_node(Node::Variable { name: "p1".to_string() }).unwrap();
    let p2_var = graph.add_node(Node::Variable { name: "p2".to_string() }).unwrap();
    let await1 = graph.add_node(Node::Await { expr: p1_var }).unwrap();
    let await2 = graph.add_node(Node::Await { expr: p2_var }).unwrap();

    // Create list: (list (await p1) (await p2))
    let list = graph.add_node(Node::List(vec![await1, await2])).unwrap();

    // Create let binding
    let let_node = graph.add_node(Node::Let {
        bindings: vec![
            ("p1".to_string(), spawn1),
            ("p2".to_string(), spawn2),
        ],
        body: list,
    }).unwrap();

    graph.root_id = Some(let_node);

    // Compile
    let compiler = Compiler::new();
    let bytecode = compiler.compile(&graph).unwrap();

    // Create VM with effect context and runtime
    let mut vm = VM::new(bytecode);
    vm.set_effect_context(context);
    vm.set_effect_runtime(runtime);

    // Run the VM (spawned tasks will execute in the background)
    let result = vm.run().expect("VM execution failed");

    // Verify result should be a list containing [1, 2]
    match result {
        Value::List(items) => {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0], Value::Integer(1));
            assert_eq!(items[1], Value::Integer(2));
        }
        _ => panic!("Expected list result, got {:?}", result),
    }
}