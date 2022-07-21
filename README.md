# [Artificial Intelligence to Classify and Detect Masquerading Users on HPC Systems from Shell Histories](https://doi.org/10.1145/3491418.3535187)

## SR-GNN
Graphs are cool and vital ways of representing information and relationships in the world around us.
Nowadays graphs are everywhere - from Social media platforms to molecules in pharmaceutical settings. The famous Konigsberg problem can be solved using graphs.

We can ask some amazing questions from the relevant graph data - for instance - Does this person smoke? What is the next movie I should watch? Is this molecule a suitable drug?

So the question here is how can we train a machine learning algorithm to learn the patterns in a shell history of a user so that later we can identify similar behaviour for different users?
We generated input data such that each command sequence has the next command as its target label. Then, we ask the model what is the next command given a sequence of commands.
In graphical terms, for the shell histories, each command is a node, and edges embed information on when these commands were issued.

![Problem definition](https://github.com/jackfrost1411/masquerade-detection-using-srgnn/blob/main/images/Problem.png)

Now we want to learn a “neural network suitable representation” of the graph data also known as Representation learning.
Graph neural networks are the way to do this.

![Two Users](https://github.com/jackfrost1411/masquerade-detection-using-srgnn/blob/main/images/Two%20users.png)
This is how two different users would look like to a GNN after training. X-axis are the learned feature representations of the commands on y axis. This features define user behaviour.
The overall architecture is visualized in the paper (https://doi.org/10.1145/3491418.3535187). We pass in the sequence of commands from a user’s shell history as graphs to a GNN and get the final command embeddings. We apply a attention mechanism to ensure that we rely on the entire session to predict the next command and not just the last executed command. At last, we have a softmax function to predict the probability scores for each command in dictionary being the next command.

How do we run inference on the model trained on one user’s shell history? We pass in random command sequences of various test users in the batch of 200 to this model and ask the model to predict the next command in the hope that model will recognize the similar command patterns - if the sequence is of the same user the model is trained on then it will result in high accuracy and vice versa.

We were able to find the aliases of the user that the model was trained on. The paper contains intersesting visualizations on the achived outcomes.
