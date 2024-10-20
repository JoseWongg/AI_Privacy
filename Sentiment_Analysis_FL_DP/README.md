The project involves training an LSTM model for binary sentiment analysis classification using a dataset containing 21966 reviews with six columns, each representing different attributes of the reviews: 

name: The reviewer's name (string).
Example: "John Doe"

country: The reviewer's country (string).
Example: "US"

date_time: The date the review was posted (string).
Example: "Jan 12, 2023"

stars: The rating the reviewer gives (on a scale of 1 to 5).
Example: 4

review_head: The headline of the review (string).
Example: "Great experience overall"

review_body: The detailed text of the review (string).
Example: "I had a great experience using this service. The platform is user-friendly and customer support was responsive. Highly recommend!"

The project utilizes the TensorFlow Federated package to simulate an iterative federated learning process. During each iteration, the LSTM model is sent to five clients to be trained on their local data. After each iteration, the edge models from each client are collected and aggregated to update the global model. This updated global model is then redistributed for further training in the next iteration.

Additionally, the LSTM model used by the clients incorporates a custom differential privacy optimizer, which applies gradient clipping and noise addition techniques to protect the privacy of client data during the federated learning process. After training is complete, the final Tensorflow-Federated model's weights are transferred to a Keras LSTM for general use. 

Privacy budget and performance are evaluated.
