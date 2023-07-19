# Grimoire

"We call ourselves modelers. What we essentially do is to pay very little attention to what people say they do and a great deal of attention to what they do. And then we build ourselves a model of what they do. We are not not psychologists, and we're not theologians or theoreticians. We have no idea about the "real" nature of things, and we're not particularly interested in what's "true." The function of modeling is to arrive at descriptions which are useful. So, if we happen to mention something that you know from scientific study, or from statistics, is inaccurate, realize that a different level of experience is being offered you here. We're not offering you something that's true, just things that are useful."

Quote from wizards John Grinder and Richard Bandler taken from the 'grimoire' Frogs Into Princes

Grimoire is an NLP (Natural Language Processing) framework designed to simplify the process of creating a corpus of authentic text and engineering features. Our goal is to abstract away the complex tasks, allowing you to focus on what really matters: crafting high-quality machine learning models.

Through Grimoire, creating a corpus ready to train a machine learning model is as easy as providing a list of document ID numbers. We handle the rest!

Features
Document Fetching: Provide us with a list of document ID numbers, and Grimore fetches all necessary documents.
Text Preprocessing: Automatic removal of noise, such as special characters, numbers, extra spaces, etc.
Tokenization: Built-in tokenization for ease of text segmentation.
Feature Engineering: Automatic generation of necessary NLP features such as TF-IDF, Word2Vec, etc.
Customizability: Allows for customization of preprocessing, tokenization and feature generation steps.
Efficiency: Highly efficient and fast processing with parallel and batch processing capabilities.
Scalability: Built with scalability in mind. Grimoire can handle from small to large data sets.
Installation
To install Grimoire, simply run the following pip command:

Copy code
pip install grimoire
Basic Usage
python
Copy code
from grimoire.connectors import DoclinkConnector
from grimoire.pipeline import Corpus

# List of document ids
document_ids = ['123', '456', '789']

# Initialize builder
connector = DoclinkConnector()
corpus = Corpus(connector)

# Build the corpus
documents = corpus.add_documents(document_ids)

# To get feature vectors
features = corpus.get_features()
Documentation
For more details on how to use Grimoire and information on advanced topics like customization of preprocessing, tokenization and feature generation steps, please check our Documentation.

Contribute
We would love your help to make Grimoire better! If you're interested in contributing, please check out our Contributing Guide.

Support
If you encounter any issues or require further assistance, please file an issue on the Grimoire GitHub issue tracker.

License
Grimoire is licensed under the MIT License.
