import tensorflow as tf
import keras
from keras import layers

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve

def positional_encoding(length, depth):
    depth = depth/2
    
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    
    angle_rates = 1/(10000**depths)
    angle_rads = positions * angle_rates
    
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )
    
    return tf.cast(pos_encoding, dtype=tf.float32)

#  Token Emebdding Layer and Positional Encoding
class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size, emb_dim, max_len, dropout = None, regularizer = None):
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.token_emb = layers.Embedding(
                self.vocab_size, self.emb_dim, mask_zero=True, embeddings_regularizer = regularizer
        )
        self.pos_enc = positional_encoding(self.max_len, self.emb_dim)
        
        self.dropout = dropout
        if self.dropout is not None:
            self.dropout_layer = layers.Dropout(self.dropout)
    
    def compute_mask(self, *args, **kwargs):
        return self.token_emb.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        token_emb = self.token_emb(x)
        token_emb *= tf.math.sqrt(tf.cast(self.emb_dim, tf.float32))
        token_emb = token_emb + self.pos_enc[tf.newaxis, :length, :]
        
        if self.dropout is not None:
            return self.dropout_layer(token_emb)
        else:
            return token_emb
        
class Encoder(layers.Layer):
    def __init__(
            self,
            vocab_size,
            maxlen,
            emb_dim,
            num_heads,
            ffn_dim,
            dropout=0.1,
            regularizer = None
        ):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attention = None
        self.regularizer = regularizer
        
        # In most of the Attention implementation the query, key and value layer do not have biased added 
        # even in formula we just multipy with the weights and do not add bias.
        self.attn = layers.MultiHeadAttention(self.num_heads, self.emb_dim, use_bias=False, kernel_regularizer=self.regularizer)
        self.ffn_layer = keras.Sequential([
            layers.Dense(self.ffn_dim, activation='relu', kernel_regularizer=self.regularizer),
            layers.Dropout(self.dropout),
            layers.Dense(self.emb_dim, kernel_regularizer=self.regularizer)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout)
        self.dropout2 = layers.Dropout(self.dropout)


    def call(self, x):
        attn_output = self.attn(query=x, key=x, value=x, use_causal_mask = True)
        x = self.layernorm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn_layer(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))

        return x

@keras.saving.register_keras_serializable()
class Transformer(keras.Model):
    def __init__(
            self,
            vocab_size,
            maxlen,
            emb_dim,
            num_heads,
            ffn_dim,
            num_classes,
            num_layers = 1,
            dropout = 0.1,
            regularizer = None
    ):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.emb_dim = emb_dim
        self.maxlen = maxlen
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.regularizer = regularizer

        self.token_emb = TokenEmbedding(self.vocab_size, self.emb_dim, self.maxlen, self.dropout, self.regularizer)

        self.encoder_stack = keras.Sequential([
            Encoder(self.vocab_size, self.maxlen, self.emb_dim, self.num_heads, self.ffn_dim, self.dropout, self.regularizer)
            for _ in range(self.num_layers)
        ])

        self.average_pool = layers.GlobalAveragePooling1D()
        self.dropout_layer = layers.Dropout(self.dropout)
        self.clf_head = layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=self.regularizer)

    def call(self, x):
        x = self.token_emb(x)
        
        x = self.encoder_stack(x)
        x = self.average_pool(x)
        x = self.dropout_layer(x)
        probs = self.clf_head(x)

        return probs

    # Tooked reference my Deep learning Week-5 Assignment
    def visualize_model(self, history):
        plt.figure(figsize=(14, 6))
        # Extract the metrics to visulalize
        metrics = []

        # Getting all the metrics we have while model training
        hist_metrics = history.history.keys()
        for item in hist_metrics:
            if item.startswith("val"):
                continue

            metrics.append(item)

        for indx, metric in enumerate(metrics):
            title = f'{metric}'
            legends = [metric]
            plt.subplot(1, 2, indx+1)
            plt.plot(history.history[metric], label=metric, marker='o')

            val_metric = 'val_' + metric
            if val_metric in hist_metrics:
                title += f" vs {val_metric}"
                plt.plot(history.history[val_metric], label=val_metric, marker='^')
                legends.append(val_metric)

            plt.legend(legends)
            plt.title(title)

        plt.show()

    def preds(self, dataset: tf.data.Dataset):
        y_true = []
        y_pred = []

        dataset_len = len(dataset)
        for inp, label in dataset.take(dataset_len):
            pred = self.call(inp).numpy()
            y_true.extend(label.numpy())
            y_pred.extend(pred)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true_label = np.argmax(y_true, axis=-1)
        y_pred_label = np.argmax(y_pred, axis=-1)

        return y_true, y_true_label, y_pred, y_pred_label

    def plot_confusion_matrix(self, conf_matrix, labels):
        plt.figure(figsize=(8, 6))
        plt.title("Confusion Matrix", {'size': 14})
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted", {'size': 12})
        plt.ylabel("Actual", {'size': 12})
        plt.show()

    def plot_roc_curve(self, y_true, y_pred, labels):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[label] = auc(fpr[label], tpr[label])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(6, 6))
        plt.title("ROC Curve", {'size': 14})
        plt.plot(fpr["micro"], tpr["micro"], label=f"ROC micro-avg area({roc_auc['micro']*100:.1f}%)")

        for label in labels:
            plt.plot(fpr[label], tpr[label], label=f"ROC {label} area({roc_auc[label]*100:.1f})%")

        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid()
        plt.legend(loc="lower right")
        plt.show()
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "emb_dim": self.emb_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "regularizer": self.regularizer
        }
        
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        vocab_size = config.pop("vocab_size")
        maxlen = config.pop("maxlen")
        emb_dim = config.pop("emb_dim")
        num_heads = config.pop("num_heads")
        ffn_dim = config.pop("ffn_dim")
        num_classes = config.pop("num_classes")
        num_layers = config.pop("num_layers")
        dropout = config.pop("dropout")
        regularizer = config.pop("regularizer")
        
        return cls(vocab_size, maxlen, emb_dim, num_heads, ffn_dim, num_classes,
                  num_layers, dropout, regularizer)

def get_model(filepath):
    return keras.models.load_model(filepath)

if __name__ == "__main__":
    reg = keras.regularizers.L1(l1=1e-5)
    model = Transformer(
        vocab_size = 2000,
        maxlen = 32,
        emb_dim = 32,
        num_heads = 2,
        ffn_dim = 32,
        num_classes = 3,
        num_layers = 1,
        dropout = 0.5,
        regularizer = reg
    )

    model.load_weights('finetune_model1.weights.h5')