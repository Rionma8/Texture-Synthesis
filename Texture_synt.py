from importlib.resources import path
import numpy as np
import cv2 as cv
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

#POW = 2
POW = 0.9

#In case we want gauss weights on each neighborhood
GAUSS = True


def Center_positions(im_size,w,e):
	"""
	im_size : taille de l'image (im_size,im_size,3)
	w : taille des voisinages
	e : espace entre les centres de 2 voisinages successifs
	return : Positions des Centres des voisinages sur la première ligne
	"""
	N_center = (im_size - (w//2)*2+ e)//(1+e) #Nombre de centres sur une ligne
	remaining_pix = (im_size - (w//2)*2 + e)%(1+e) #pixels qui ne sont pas dans un voisinage au bout de la ligne
	Centers = np.full(N_center,1+e) #On remplir une liste avec 1+e
	Centers[0] = w//2  #le premier centre est situé à w//2
	Centers[-remaining_pix:] += 1  #On décale les derniers centre pour que tous les pixels appartiennent à un voisinage
	return np.cumsum(Centers) #somme cumulée pour avoir la position des centres

def create_overlap_mask(X_indices,w,X_size,Distances,gauss_weights):
	"""
	X_indices : indices des pixels des voisinages (nombre de voisinages x nombre de pixels par voisinage)
	w : taille des voisinages
	X_size : taille de l'image X
	Distances : Distances entre zp et xp
	return : un masque remplit avec les poids qui servira pour la E_step
	"""
	weights = np.repeat(Distances**(POW-2),3*w**2) 
	weights *= gauss_weights
	flat_Mask = np.bincount(X_indices.flat,weights,minlength=3*X_size**2)
	Mask = np.reshape(flat_Mask,(X_size,X_size,3))
	return Mask

def E_step(Z_p,Z_Patches,X_size,X_indices,Distances,Mask,gauss_weights):
	"""
	Z_p : indique les voisinages de Z les plus proches des voisinages de X
	Z_Patches : valeur des pixels pour chaque voisinage de Z (nombre de voisinages x nombre de pixels par voisinage)
	X_size : taille de l'image X
	X_indices : indices des pixels des voisinages (nombre de voisinages x nombre de pixels par voisinage)
	Distances : Distances entre zp et xp
	Mask : Masque obtenu avec create_overlap_mask
	return : L'image X après la E_step
	"""
	weights = (np.reshape(Distances**(POW-2),(-1,1))*Z_Patches[Z_p]).flat
	weights *= gauss_weights
	flat_X = np.bincount(X_indices.flat,weights,minlength=3*X_size**2)
	X = np.reshape(flat_X,(X_size,X_size,3))/Mask
	return X


def Patches_indices(Centers,w,im_size):
	"""
	Centers : Positions des centres des voisinages sur la première ligne de l'image
	w : taille des voisinages
	im_size : taille de l'image
	return : indices des pixels des voisinages dont les centres sont donnés par Centers
	"""
	N_center = len(Centers)
	patch_ind = np.arange(-(w//2),w//2+1)
	X_c,Y_c,d_x,d_y,color = np.meshgrid(Centers,Centers,patch_ind,patch_ind,np.arange(3))
	Indices = color + 3*(X_c + d_x + (Y_c + d_y)*im_size)
	Indices = np.reshape(Indices,(N_center**2,3*w**2))
	return Indices

def M_step(Tree,X,X_indices):
	"""
	Tree : arbre créé grâce au KDTree
	X : Texture en cours de synthèse
	X_indices : indices des pixels des voisinages (nombre de voisinages x nombre de pixels par voisinage)
	return : Les Centres des voisinages zp et les distances entre zp et xp
	"""
	X_patches = X.flat[X_indices]
	Distances, Z_p = Tree.query(X_patches)
	return Z_p[:,0],Distances


def Text_synthesis(Z_im,w,e,Init_text = None,size_text = 0,MaxIter = 50):
	"""
	Z_im : Texture originale
	w : taille des voisinages
	e : espacement entre les voisinages
	Init_text : A préciser si on ne souhaite pas partir d'une texture aléatoire
	size_text : Préciser la taille de la texture si on part d'une texture alétoire
	return : la texture synthétisée X et l'énergie à chaque itération
	"""
	
	Z_size = Z_im.shape[0]
	Z_centers = np.arange(w//2,Z_size-w//2)
	Z_Indices = Patches_indices(Z_centers,w,Z_size)
	Patches = Z_im.flat[Z_Indices]
	tree = KDTree(Patches)

	if size_text > 0:
		X_centers = Center_positions(size_text,w,e)
		Z_p = np.random.choice(Patches.shape[0],len(X_centers)**2)
		X_indices = Patches_indices(X_centers,w,size_text)
		Distances = np.ones(len(Z_p))
	else : 
		size_text = Init_text.shape[0]
		X_centers = Center_positions(size_text,w,e)
		X_indices = Patches_indices(X_centers,w,size_text)
		Z_p,Distances = M_step(tree,Init_text,X_indices)

	Energies = []

	if GAUSS == True:
		sigma_gauss = w//4
		x = np.arange(-(w//2),w//2+1)
		X,Y = np.meshgrid(x,x)
		dst = np.sqrt(X*X+Y*Y)
		gauss = np.exp(-(dst-0.0)**2 / ( 2.0 * sigma_gauss**2 ))
		gauss_weights = np.tile(gauss.flat,X_indices.shape[0]*3)
	else: 
		gauss_weights = np.ones(len(X_indices.flat))

	for t in range(MaxIter):
		#E_step
		Mask = create_overlap_mask(X_indices,w,size_text,Distances,gauss_weights)
		X = E_step(Z_p,Patches,size_text,X_indices,Distances,Mask,gauss_weights)
		Energies.append(np.sum(Distances**(POW))/len(X_indices.flat))

		#M_step
		Z_p0 = Z_p
		Z_p, Distances = M_step(tree,X,X_indices)

		if np.all(Z_p0 == Z_p) : break
	return X,Energies


def multi_level_synth(path,W,E,Res,Size_synt):
	"""Fonction pour faire de la synthèse multi-level"""
	Input_image = plt.imread(path)
	#Input_image = Input_image.astype(float)/255
	mi = min(Input_image.shape[0],Input_image.shape[1])
	Input_image = Input_image[:mi,:mi,:]
	print('Image shape : ',Input_image.shape)
	down_im2 = cv.pyrDown(Input_image)
	down_im4 = cv.pyrDown(down_im2)
	Input_images = {
		"down_im4" : down_im4,
		"down_im2" : down_im2,
		"down_im1" : Input_image
	}
	Energies = []
	Indices_reschange = [0]
	Init_texture = None
	size_text = Size_synt//Res[0]
	N_im = 1
	for i in range(len(W)):
		N_im += len(W[i])
	plt.subplots(1,N_im,figsize = (17,8))
	plot_ind = 1
	for res in range(len(Res)):
		im_input = Input_images["down_im"+str(Res[res])]
		w_res = W[res]
		e_res = E[res]
		for neigh in range(len(w_res)):
			Text_generated,Energies_text = Text_synthesis(im_input,w_res[neigh],e_res[neigh],Init_text = Init_texture,size_text = size_text,MaxIter = 50)
			Energies += Energies_text
			Indices_reschange.append(len(Energies))
			plt.subplot(1,N_im,plot_ind)
			plot_ind += 1
			plt.imshow(Text_generated.astype(float)/255)
			plt.axis('off')
			plt.title('Resolution 1/'+str(Res[res])+', w = '+str(w_res[neigh]))
			size_text = 0
			Init_texture = Text_generated
		if res < len(Res)-1 : Init_texture = cv.pyrUp(Text_generated)
		
			
	plt.subplot(1,N_im,plot_ind)
	plt.imshow(Input_image)
	plt.axis('off')
	plt.title('Input Texture')
	plt.show()
	return Text_generated, Energies,np.array(Indices_reschange[:-1])