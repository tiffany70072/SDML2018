from keras.models import Model, Sequential
from keras.layers import Input, Embedding, Flatten, merge, Concatenate, Dense, Dropout, Activation, Add
from keras import backend as K
from keras import optimizers
from keras import losses
#from IPython.display import SVG
#from keras.optimizers import Adam
#from keras.utils.vis_utils import model_to_dot

def matrix_factorization(user_dim, food_dim, latent_dim, use_food_bias = False, return_emb = False):
	user_input = Input(shape=[1], name='User')
	user_embedding = Embedding(user_dim + 1, latent_dim, name='User-Emb')(user_input)
	user_vec = Flatten(name='FlattenUsers')(user_embedding)
	#user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(user_dim + 1, n_latent_factors, name='User-Embedding')(user_input))

	food_input = Input(shape=[1], name='Food')
	food_embedding = Embedding(food_dim + 1, latent_dim, name='Food-Emb')(food_input)
	food_vec = Flatten(name='FlattenFoods')(food_embedding)
	user_embed_model = Model(user_input, user_vec)

	prod = merge([user_vec, food_vec], mode='dot', name='DotProduct')
	if use_food_bias == True:
		food_bias = Embedding(food_dim, 1)(food_input)
		food_bias = Flatten()(food_bias)
		prod = Add()([prod, food_bias])
	model = Model([user_input, food_input], prod)
	# adam = optimizers.Adam(lr=0.001)
	model.compile('adam', 'mean_squared_error')
	#model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	#SVG(model_to_dot(model,  show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
	model.summary()
	user_embed_model.summary()
	if return_emb == True: return model, user_embed_model
	else: return model 

def mf_ranking_label_loss(user_dim, food_dim, latent_dim):
	user_input1 = Input(shape=[1], name='User1')
	#user_input2 = Input(shape=[1], name="User2")
	user_shared = Sequential([Embedding(user_dim + 1, latent_dim, name='User-Emb'), Flatten(name='FlattenUsers')])
	user_vec1 = user_shared(user_input1)
	#user_vec2 = user_shared(user_input2)
	
	food_input1 = Input(shape=[1], name='Food1')
	food_input2 = Input(shape=[1], name='Food2')
	food_shared = Sequential([Embedding(food_dim + 1, latent_dim, name='Food-Emb'), Flatten(name='FlattenFoods')])
	food_vec1 = food_shared(food_input1)
	food_vec2 = food_shared(food_input2)

	rating1 = merge([user_vec1, food_vec1], mode='dot', name='DotProduct')
	rating2 = merge([user_vec1, food_vec2], mode='dot')
	x = Subtract()(rating1, rating2)
	x = Activation('sigmoid')
	model = Model([user_input1, food_input1, food_input2], x)
		
	model.compile('adam', 'binary_crossentropy')
	model.summary()
	return model
'''
def combined_loss(self, real, pred):
	loss = (losses.categorical_crossentropy(real[:, 4:], pred[:, 4:])) + \
		losses.mean_absolute_error(real[:, :4], pred[:, :4])
	print "loss =", loss
	model.compile(loss = self.combined_loss, optimizer = opt, metrics = ['mae'])
	return loss

def shared_neigh_layer(self, x):
		shared = Sequential()
		shared.add(Dense(4, activation = 'relu', input_shape=(7,), name='weight'))
		
		for i in range(self.num_neigh):
			weight = Lambda(lambda x: x[:, i, :7], output_shape=(7,))(x)
			weight = shared(weight)
			one_x = Lambda(lambda x: x[:, i, 7:], output_shape=(4,))(x)
			one_x = Multiply()([weight, one_x])
			if i == 0: concat = one_x
			else: concat = concatenate([concat, one_x], axis=-1)
		#concat = Dense(100, activation = 'relu')(concat)
		print "neigh_concat_layer (x3) =", concat.shape
		return concat
'''		


def factorization_machine(user_dim, food_dim, food_features_dim, latent_dim):
	user_input = Input(shape=[1], name='User')
	user_embedding = Embedding(user_dim + 1, latent_dim, name='User-Emb')(user_input)
	user_vec = Flatten(name='FlattenUsers')(user_embedding)
	food_input = Input(shape=[1], name='Food')
	food_embedding = Embedding(food_dim + 1, latent_dim, name='Food-Emb')(food_input)
	food_vec = Flatten(name='FlattenFoods')(food_embedding)

	food_features_input = Input(shape=[food_features_dim])
	food_vec = Dense(latent_dim)(food_features_input)

	x = Concatenate(axis = -1)([user_vec, food_vec, food_vec])
	x = Dense(128)(x)
	x = Dropout(0.5)(x)
	x = Activation('relu')(x)
	x = Dense(1)(x)
	
	model = Model([user_input, food_input, food_features_input], x)
	# adam = optimizers.Adam(lr=0.001)
	model.compile('adam', 'mean_squared_error')
	#model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	#SVG(model_to_dot(model,  show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
	model.summary()
	return model

def MLP(user_dim, food_dim, latent_dim):
	user_input = Input(shape=[1], name='User')
	user_embedding = Embedding(user_dim + 1, latent_dim, name='User-Emb')(user_input)
	user_vec = Flatten(name='FlattenUsers')(user_embedding)
	food_input = Input(shape=[1], name='Food')
	food_embedding = Embedding(food_dim + 1, latent_dim, name='Food-Emb')(food_input)
	food_vec = Flatten(name='FlattenFoods')(food_embedding)

	x = Concatenate(axis=-1)([user_vec, food_vec])
	x = Dense(128)(x)
	x = Dropout(0.5)(x)
	x = Activation('relu')(x)
	x = Dense(1)(x)
	model = Model([user_input, food_input], x)
	# adam = optimizers.Adam(lr=0.001)
	model.compile('adam', 'mean_squared_error')
	#model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	#SVG(model_to_dot(model,  show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
	model.summary()
	return model