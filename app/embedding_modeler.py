from sklearn import utils
from tqdm import tqdm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
cores = multiprocessing.cpu_count()
tqdm.pandas(desc="progress-bar")


class EmbeddingModeler:
    """Transform text dataframes into embedding dataframes and returns embedding model

    Returns:
        model -- doc2vec model 
        y_train, y_test -- tags/labels
        x_train, , x_test -- embeddings
    """
    def doc2vec_pipeline(self, train_df, test_df):
        """Transform text dataframes into embedding dataframes and returns embedding model
        
        Arguments:
            train_df {df} -- dataframe of text docs and labels
            test_df {df} -- dataframe of text docs and labels
        
        Returns:
            model -- doc2vec model 
            y_train, y_test -- tags/labels
            x_train, , x_test -- embeddings
        """
        train_tagged_docs, test_tagged_docs = tag(train_df, test_df)
        model = fit_doc2vec(train_tagged_docs)
        y_train, x_train = predict_doc2vec(model, train_tagged_docs)
        y_test, x_test = predict_doc2vec(model, test_tagged_docs)
        return model, y_train, x_train, y_test, x_test

def tag(train_df, test_df):
    """Transforms dataframe into tagged documents
    
    Arguments:
        train_df {df} -- dataframe of text docs and labels
        test_df {df} -- dataframe of text docs and labels
    
    Returns:
        train_tagged_docs, test_tagged_docs -- series of tagged documents with text and label
    """
    print('document tagging')
    train_tagged_docs = train_df.apply(
                                        (lambda r: TaggedDocument((r['features']),
                                        tags=[r['response']])), axis=1)

    test_tagged_docs = test_df.apply(
                                        (lambda r: TaggedDocument((r['features']),
                                        tags=[r['response']])), axis=1)
    return train_tagged_docs, test_tagged_docs

def fit_doc2vec(train_tagged_docs):
    """Take in tagged documents and trains doc2vec model
    
    Arguments:
        train_tagged_docs {series} -- series of tagged documents with text and label 
    
    Returns:
        model -- trained doc2vec model 
    """
    model = Doc2Vec(dm=1,
                    negative=5,
                    vector_size=100,
                    random_state=42,
                    workers=cores,
                    min_count=2)

    model.build_vocab([x for x in tqdm(train_tagged_docs)])

    for epoch in range(2):
        model.train(utils.shuffle([x for x in tqdm(train_tagged_docs)]),
                    total_examples=len(train_tagged_docs),
                    epochs=epoch)
    return model

def predict_doc2vec(model, tagged_docs, steps=20):
    """Input model and create vectors for documents
    
    Arguments:
        model {model} -- doc2vec model
        tagged_docs {series} -- series of text documents 
    
    Keyword Arguments:
        steps {int} -- number of epochs ran in prediction (default: {20})
    
    Returns:
        tags -- list of document tags/labels
        vectors -- list of vectors embeddings for documents
    """
    tags = [td.tags for td in tagged_docs]
    vectors = [model.infer_vector(td.words, steps=steps)
                for td in tagged_docs]
    return tags, vectors
