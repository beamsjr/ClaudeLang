use fluentai_core::ast::{Graph, Literal, Node};
use fluentai_core::value::Value;
use fluentai_vm::{compiler::Compiler, VM};

fn main() {
    // Test simple variable capture in let expression
    let mut graph = Graph::new();

    // Create a simple value
    let val = graph.add_node(Node::Literal(Literal::Integer(42))).unwrap();

    // Create a lambda that references the variable x
    let x_var = graph.add_node(Node::Variable { name: "x".to_string() }).unwrap();
    let lambda = graph.add_node(Node::Lambda {
        params: vec![],
        body: x_var,
    }).unwrap();

    // Create let binding: (let ((x val)) lambda)
    let let_node = graph.add_node(Node::Let {
        bindings: vec![("x".to_string(), val)],
        body: lambda,
    }).unwrap();

    graph.root_id = Some(let_node);

    // Compile
    let compiler = Compiler::new();
    let bytecode = compiler.compile(&graph).unwrap();

    println!("Compiled successfully!");
    println!("Bytecode: {:?}", bytecode);

    // Create VM
    let mut vm = VM::new(bytecode);

    // Run the VM
    let result = vm.run().expect("VM execution failed");

    println!("Result: {:?}", result);
    println!("Expected: {:?}", Value::Integer(42));
    
    // Check if the lambda captured the variable correctly
    if result == Value::Integer(42) {
        println!("SUCCESS: Variable capture works!");
    } else {
        println!("FAILURE: Variable capture failed!");
    }
}