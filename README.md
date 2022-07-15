# Masquerade Detection Using SR-GNN

Graphs are cool and vital ways of representing information and relationships in the world around us.
Nowadays graphs are everywhere - from Social media platforms to molecules in pharmaceutical settings. The famous Konigsberg problem can be solved using graphs.

We can ask some amazing questions from the relevant graph data - for instance - Does this person smoke? What is the next movie I should watch? Is this molecule a suitable drug?

So the question here is how can we train a machine learning algorithm to learn the patterns in a shell history of a user so that later we can identify similar behaviour for different users?
We generated input data such that each command sequence has the next command as its target label. Then, we ask the model what is the next command given a sequence of commands.
In graphical terms, for the shell histories, each command is a node, and edges embed information on when these commands were issued.

Now we want to learn a “neural network suitable representation” of the graph data also known as Representation learning.
But,
1) Neural networks expect fixed number of inputs
2) We can flip the graph it’s still the same just the order of the nodes change, we can’t input adjacency matrix as input to the feed forward neural network as it is sensitive to changes in the node order.
So we need a algorithm to be size independent and permutation invariant.

Graph neural networks are the way to do this.
We have a set of initial node embeddings and we pass this information to the GNN and get a new set of embeddings for each node.
Similar nodes (nodes with similar features or structural context will lead to similar node embeddings). This node embeddings cannot be interpreted as it’s a artificial compound of node and edge features within the graph.

Graph neural networks have several message passing layers. But what happens in each layer?

Consider this example graph - first of all a message from its neighbors is prepared depending on the type of edges. 
This prepared message is then summarized by simple summation or max pooling operation. 
Finally, next state of the node is computed from its previous state and the message from its neighbors.

![Two Users](https://github.com/jackfrost1411/masquerade-detection-using-srgnn/blob/main/images/Two%20users.png)
This is how two different users would look like to a GNN after training. X-axis are the learned feature representations of the commands on y axis. This features define user behaviour.
Here, is the overall architecture we employed in our paper. We pass in the sequence of commands from a user’s shell history as graphs to a GNN and get the final command embeddings. We apply a attention mechanism to ensure that we rely on the entire session to predict the next command and not just the last executed command. At last, we have a softmax function to predict the probability scores for each command in dictionary being the next command.

How do we run inference on the model trained on one user’s shell history? We pass in random command sequences of various test users in the batch of 200 to this model and ask the model to predict the next command in the hope that model will recognize the similar command patterns - if the sequence is of the same user the model is trained on then it will result in high accuracy and vice versa.

In the effort of detecting masquerade on a supercomputer, two methods are successfully applied. Transformers take more time in training but run faster in detecting the similar users shell histories. Whereas the GNN is faster at training. Transformer can look at users globally and identify group of similar users and GNN then can be employed to verify and find the aliases from that group. In particular for the Transformer, the prompts were of size 400 that means it needs at least few hundred commands to train. How many minimally sized prompts (and what is the minimum size) remain open questions.
Thank you!
