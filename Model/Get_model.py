
from Useful_layers import *
#from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer

#from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, Dense, Input, LSTM, Embedding, Dropout, Activation,GlobalMaxPooling1D,CuDNNGRU,CuDNNLSTM,GlobalAveragePooling1D,Conv1D,Conv2D,Flatten,Reshape,MaxPool2D,PReLU,add,MaxPooling1D
from keras.layers.merge import Concatenate
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.callbacks import Callback
from keras.layers.core import SpatialDropout1D
import pickle
########################################
## set directories and parameters
########################################

from keras.engine import Layer, InputSpec
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints



def get_CNN2d(embedded_sequences,embed_size,MAX_SEQUENCE_LENGTH,Para_dic):

    ### CNN2d 
    CNN_x = Reshape((MAX_SEQUENCE_LENGTH, embed_size, 1))(embedded_sequences)
    filter_sizes = Para_dic['CNN2d_filter_sizes']
    num_filters = Para_dic['CNN2d_num_filters']
    
    conv_2d_feature=[]
    for size in filter_sizes:
        conv=Conv2D(num_filters, kernel_size=(size, embed_size), kernel_initializer='he_uniform', activation='elu')(CNN_x)
    
        maxpool= MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - size + 1, 1))(conv)

        conv_2d_feature.append(maxpool)

    z = Concatenate(axis=1)(conv_2d_feature)   
    CNN2d = Flatten()(z)
    return CNN2d

## credit to https://www.kaggle.com/michaelsnell/conv1d-dpcnn-in-keras
def get_DPCNN(embedded_sequences,Para_dic):
    ### DPCNN

    filter_nr = Para_dic['DPCNN_filter_nr']
    filter_size = Para_dic['DPCNN_filter_size']
    max_pool_size = Para_dic['DPCNN_max_pool_size']
    max_pool_strides = Para_dic['DPCNN_max_pool_strides']
    train_embed = False

    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(embedded_sequences)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    #we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
    #if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
    resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(embedded_sequences)
    resize_emb = PReLU()(resize_emb)
    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
        
    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
        
    block3_output = add([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3_output)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block4)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)

    DPCNN = add([block4, block3_output])
    DPCNN = GlobalMaxPooling1D()(DPCNN)

    return DPCNN


def get_model(embedding_layer,RNN,embed_size,Feature_dic,Para_dic):

    MAX_SEQUENCE_LENGTH=Para_dic['MAX_SEQUENCE_LENGTH']
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_raw= embedding_layer(comment_input)
    embedded_sequences = SpatialDropout1D(Para_dic['spatial_dropout'])(embedded_sequences_raw)
    
    ### RNN
    if RNN=='LSTM':
        RNN_x = Bidirectional(CuDNNLSTM(Para_dic['num_lstm'],return_sequences=True))(embedded_sequences)
    elif RNN=='GRU':
        RNN_x = Bidirectional(CuDNNGRU(Para_dic['num_lstm'],return_sequences=True))(embedded_sequences)

    Feature=[]

    ######## RNN Features
    ##### Attention
    if Feature_dic['Attention']==1:
        Feature.append(Attention(MAX_SEQUENCE_LENGTH)(RNN_x))

    if Feature_dic['RNN_maxpool']==1:
        Feature.append(GlobalMaxPooling1D()(RNN_x))

    ##### Capsule
    if Feature_dic['Capsule']==1:

        capsule = Capsule(share_weights=True)(RNN_x)
        capsule = Flatten()(capsule)
        Feature.append(capsule)

    ##### RNN_CNNN1d

    if Feature_dic['RNN_CNN_conv1d']==1:
        
        Cx = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(RNN_x)
        avg_pool = GlobalAveragePooling1D()(Cx)
        max_pool = GlobalMaxPooling1D()(Cx)
        Feature.append(avg_pool)
        Feature.append(max_pool)

    ######## CNN Features
    ### CNN2d
    if Feature_dic['CNN2d']==1:
        CNN2d=get_CNN2d(embedded_sequences,embed_size,MAX_SEQUENCE_LENGTH,Para_dic)
        Feature.append(CNN2d)


    ### DPCNN
    if Feature_dic['DPCNN']==1:
        DPCNN=get_DPCNN(embedded_sequences,Para_dic)
        Feature.append(DPCNN)


    ### Concatnation
    merged = Concatenate()(Feature)

    ### dense, add L1 reg to enable sparsity
    merged = Dense(Para_dic['dense_num'], \
                   activation=Para_dic['dense_act'],\
                   kernel_regularizer=regularizers.l1(Para_dic['L1_reg']))(merged)

    merged = Dropout(Para_dic['dense_dropout'])(merged)
    preds = Dense(6, activation='sigmoid')(merged)

    model = Model(inputs=[comment_input],         outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy'])
    print(model.summary())
    return model



 

