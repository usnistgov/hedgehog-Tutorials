---
layout: "page"
title: Moving from Hedgehog v.2 to Hedgehog v.3
---

There are few differences between Hedgehog v.2 and Hedgehog v.3.
This page is made to give you few pointers if you encounter some difficulties when porting your project to the newest version of the library. 
If your problem is not covered in these few points, then feel free to contact us.

# Content
- [Template arguments definition](#template-arguments-definition)
- [Graph API](#graph-api)
- [Can terminate API](#can-terminate-api)
- [State Manager / State](#state-manager--state)
- [Memory manager](#memory-manager)
- [Execution pipeline](#execution-pipeline)

----

# Template arguments definition

The biggest change between Hedgehog v.2 and Hedgehog v.3 is the addition of multiple outputs.
A node (a task or a graph for example), can now have multiple input types and multiple output types.

To express this change, the template argument list has changed. 

Before the API looked like this: 
```c++
Node<Output, Input1, Input2, Input3>
```

This defines a node that accepts values of type *Input1*, *Input2*, and *Input3* and sends values of type *Output*.

The new API now looks like this: 
```c++
Node<Separator, Type1, Type2, Type3, ...>
```

The *Separator* marks the number of inputs for the node, followed the input types. After the *Separator*th input, then the remaining types are interpretted as output types.

For example, if a task is defined in Hedgehog v.2 as follows:
```c++
class IntDoubleCharToFloat : public hh::AbstractTask<float, int, double, char>
```

It is now defined in Hedgehog v.3 : 
```c++
class IntDoubleCharToFloat : public hh::AbstractTask<3, int, double, char, float>
```
It reads: the three first types (*int*, *double*, and *char*) are the input types of the task, and the task has only one output type *float*.

The graph: 
```c++
class NewGRaph : public hh::Graph<3, A, B, C, D, E, F>
```
is a graph that accepts the types *A*, *B*, and *C* and produces the types *D*, *E*, and *F*. This configuration is not expressible in Hedgehog v.2.

----

# Graph API

Because of the addition of multiple outputs, the connections in a graph are slightly different. 
In Hedgehog v.2, there were 3 methods to create edges in Hedgehog: 
```c++
template<HedgehogMultiReceiver UserDefinedInput>
void input(std::shared_ptr<UserDefinedInput> input); // To set a node as input of the graph for all common types

template<HedgehogSender UserDefinedOutput>
void output(std::shared_ptr<UserDefinedOutput> output); // To set a node as output of the graph for all common types
  
template<HedgehogSender UserDefinedSender, HedgehogMultiReceiver UserDefinedMultiReceiver>
void addEdge(std::shared_ptr<UserDefinedSender> from, std::shared_ptr<UserDefinedMultiReceiver> to); // To draw an edge between two nodes for the only possible common type
```
In the newest API, we have added additional methods to handle the various types of inputs and outputs. One mechanism to accept all valid inputs and outputs between nodes and another specify the specific input/output. 

```c++

// Set a node of type InputNode_t as input of the graph for the type InputDataType
template<
    class InputDataType, 
    tool::CompatibleInputNodeForAType<InputDataType, typename core::GIM<Separator, AllTypes...>::inputs_t> InputNode_t
>
void input(std::shared_ptr<InputNode_t> inputNode);

// Set a node of type OutputNode_t as output of the graph for the type OutputType
template<
    class OutputType,
    tool::CompatibleOutputNodeForAType<OutputType, typename core::GOM<Separator, AllTypes...>::outputs_t> OutputNode_t
>
void output(std::shared_ptr<OutputNode_t> outputNode);

// Create an edge between the nodes of type SenderNode_t and ReceiverNode_t for the type CommonType
template<
    class CommonType,
    tool::SenderNodeForAType<CommonType> SenderNode_t, tool::ReceiverNodeForAType<CommonType> ReceiverNode_t
>
void edge(std::shared_ptr<SenderNode_t> sender, std::shared_ptr<ReceiverNode_t> receiver);

// Set a node of type InputNode_t as input of the graph for all the common types between the node's input types and the graph's input types
template<tool::CompatibleInputNode<typename core::GIM<Separator, AllTypes...>::inputs_t> InputNode_t>
void inputs(std::shared_ptr<InputNode_t> inputNode);

// Set a node of type OutputNode_t as output of the graph for all the common types between the node's output types and the graph's output types
template<tool::CompatibleOutputNode<typename core::GOM<Separator, AllTypes...>::outputs_t> OutputNode_t>
void outputs(std::shared_ptr<OutputNode_t> outputNode);

// Create an edge between the nodes of type SenderNode_t and ReceiverNode_t all the common types between the sender's output types and the receiver's input types
template<tool::SenderNode SenderNode_t, tool::ReceiverNode ReceiverNode_t>
void edges(std::shared_ptr<SenderNode_t> sender, std::shared_ptr<ReceiverNode_t> receiver);
```

----

# Can terminate API

The *canTerminate* method used in nodes to stop the computation early is now const. 
The signature was:
```cpp
virtual bool canTerminate();
```
and now is: 
```cpp
[[nodiscard]] virtual bool canTerminate() const;
```
----

# State Manager / State

The state manager constructor has slightly changed, because a *hh::AbstractState* is mandatory to create a state manager, it has been put as the first constructor argument.  

The constructor setting the name and the state has been changed from:


```cpp
StateManager(std::string_view const name,  // Name
           std::shared_ptr<StateType> const state, // State 
           bool automaticStart = false); // Set the state manager to not start with a nullptr data 
```

to: 


```cpp
explicit StateManager(
  std::shared_ptr<AbstractState<Separator, AllTypes...>> state, // State 
  std::string const &name = "State manager", // Name [default="State manager"] 
  bool const automaticStart = false); // Set the state manager to not start with a nullptr data 
```

Furthermore, the hh::AbstractState method to send data to the successor node[s] has been changed from *push()* to *addResult()* to be more coherent with the rest of the library. 

----

# Memory manager

The memory manager /  managed data has been changed to be more explicit. 

## ManagedMemory / MemoryManager

The *MemoryData* has been renamed *ManagedMemory* by symmetry to the *MemoryManager* name. 

To create a *ManagedMemory*, before we use a CRTP constructs: 
```cpp
template<class ManagedMemory>
class MemoryData : public std::enable_shared_from_this<MemoryData<ManagedMemory>>;

// which translate to 

class A : public hh::MemoryData<A>;

// When creating a type A usable by a MemoryManager
```

Now, to create a type actionable by a *MemoryManager* a simple inheritance is enough: 
```cpp
class A : public hh::ManagedMemory;
```

This change is due to the relaxation of the rule saying that the type of the managed memory attached to a node should be the same as a node's output type. 
Now the type managed could be any type (at the moment it inherits from *hh::ManagedMemory*). 
The counterpart is, the *MemoryManager* returns a *std::shared_ptr<ManagedMemory>* when *getManagedMemory()* is used in compatible nodes (and not the *real* type as before, however *std::dynamic_pointer_cast* can be used to recover it: *auto a = std::dynamic_pointer_cast\<A\>(this->getManagedMemory())*). 

## API changes

| **Hedgehog v.2** | **Hedgehog v.3** | **Documentation**                                                                                                                                                                     |
|------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      used()      |   postProcess()  | Mechanism called by Hedgehog when the node returns the memory before it is tested for being recycled (for example in the case the ManagedMemory is returned to the MemoryManager multiple times before being recycled (based on return of canBeRecycled) and sent back to the Pool). |
|     recycle()    |      clean()     | Clean the ManagedMemory, the method is called right before sending it back to the pool in order to help having consistent data in the pool at any time.                               |
|      reuse       |   preProcess()   | Pre-process the data when acquired through getManagedMemory() to do specific allocation or data setup.                                                                                |

----

# Execution pipeline

The constructor for the execution pipeline has been rethought to be simpler to use. 
There are now two constructors: 
```cpp

AbstractExecutionPipeline(
  std::shared_ptr<Graph<Separator, AllTypes...>> graph,
  size_t const &numberGraphs,
  std::string const name = "Execution pipeline");

AbstractExecutionPipeline(
  std::shared_ptr<Graph<Separator, AllTypes...>> const graph,
  std::vector<int> const &deviceIds,
  std::string const name = "Execution pipeline")
      
```

For both, the first argument is the graph that will be duplicated and the last is the execution pipeline's name. 
The second argument is:
  - numberGraphs: The number of graphs in the execution pipeline. The device ids are automatically generated in sequence and attached to the graphs. So, if *numberGraphs = 5*, there are 4 clones + the base graph, and the device ids generated are 0, 1, 2, 3, and 4.
  - deviceIds: The devices ids attached to the graphs in the execution pipeline. The number of graphs in the execution pipeline is deduced from the number of given device ids. If the given vector is [4, 5, 6], the base graph is cloned twice (for a total of 3 graphs), and the device ids 4, 5, and 6 are attached to each of the graphs. 
