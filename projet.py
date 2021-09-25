##############################################################################
# ------------------- PROJET ML: RESEAU DE NEURONES DIY -------------------- #
##############################################################################


""" L'objectif de ce projet est de développer une librairie python
    implémentant un réseau de neurones. L'implémentation est inspirée des
    anciennes versions de pytorch (avant l'autograd) et des implémentations
    analogues qui permettent d'avoir des réseaux génériques très modulaires.

    Chaque couche du réseau est vu comme un modume, et un réseau est constitué
    d'un ensemble de modules.
    En particulier, les fonctions d'activation sont aussi considérées comme
    des modules.
"""


################### IMPORTATION DES LIBRAIRIES ET MODULES ####################

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time

from mltools import *
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE            # Import TSNE capability from scikit


########################### CLASSE ABSTRAITE LOSS ############################

class Loss(object):
    """ Classe abstraite pour le calcul du coût.
        Note: y et yhat sont des matrices de taille batch × d : chaque
             supervision peut être un vecteur de taille d, pas seulement un
             scalaire comme dans le cas de la régression univariée.
    """
    def forward(self, y, yhat):
        """ Calcule le coût en fonction des deux entrées.
            @param y: (float) array x array, supervision
            @param yhat: (float) array x array, prédiction
            @return : (float) array, vecteur de dimension batch (le nombre d'
                       exemples).
        """
        pass

    def backward(self, y, yhat):
        """ Calcule le gradient du coût par rapport à yhat.
            @param y: (float) array x array, supervision
            @param yhat: (float) array x array, prédiction
        """
        pass


class MSELoss(Loss):
    """ Classe pour la fonction de coût aux moindres carrés.
    """
    def forward(self, y, yhat):
        """ Calcul du coût aux moindres carrés (mse).
            @return : (float) array, coût de taille batch.
        """
        return np.linalg.norm( y-yhat, axis=1) ** 2

    def backward(self, y, yhat):
        """ Calcule le gradient du coût aux moindres carrés par rapport à yhat.
            @return : (float) array x array, gradient de taille batch x d.
        """
        return -2 * (y-yhat)


class CE(Loss):
    """ Classe pour la fonction de coût cross-entropique.
    """
    def forward (self, y, yhat) :
        """ Calcul du coût cross-entropique.
            @return : (float) array, coût de taille batch.
        """
        return - np.sum( y * yhat , axis = 1 )

    def backward (self, y, yhat) :
        """ Calcule le gradient du coût cross-entropique par rapport à yhat.
            @return : (float) array x array, gradient de taille batch x d.
        """
        return - y


class CESM(Loss):
    """ Classe pour la fonction de coût cross-entropique appliqué au log SoftMax.
    """
    def forward (self, y, yhat) :
        """ Calcul du coût cross-entropique appliqué au log SoftMax.
            @return : (float) array, coût de taille batch.
        """
        return - np.sum( y * yhat , axis = 1 ) + np.log( np.sum( np.exp(yhat), axis = 1 ) )

    def backward (self, y, yhat) :
        """ Calcule le gradient du coût cross-entropique par rapport à yhat.
            @return : (float) array x array, gradient de taille batch x d.
        """
        s = SoftMax().forward( yhat )
        return - y + s * ( 1 - s )


class BCE (Loss) :
    """ Classe pour la fonction de coût cross-entropique binaire.
    """
    def forward (self, y, yhat):
        return -( y * np.maximum( -100, np.log( yhat + 0.01 ) ) + ( 1 - y ) * np.maximum( -100, np.log( 1 - yhat + 0.01 ) ) )

    def backward (self, y, yhat) :
        return - ( ( y / ( yhat + 0.01 ) )- ( ( 1 - y ) / ( 1 - yhat + 0.01 ) ) )


########################## CLASSE ABSTRAITE MODULE ###########################

class Module(object):
    """ Classe abstraite représentant un module générique du réseau de
        neurones. Ses attributs sont les suivants:
            * self._parameters: obj, stocke les paramètres du module, lorsqu'il
            y en a (ex: matrice de poids pour un module linéaire)
            * self._gradient: obj, permet d'accumuler le gradient calculé
    """
    def __init__(self):
        """ Constructeur de la classe Module.
        """
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """ Réinitialise le gradient à 0.
        """
        pass

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
        """
        pass

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        """ Met a jour la valeur du gradient: calcule le gradient du coût par
            rapport aux paramètres et l'additionne à la variable self._gradient
            en fonction de l'entrée input et des δ de la couche suivante (delta).
            @param input
            @param delta
        """
        pass

    def backward_delta(self, input, delta):
        """ Calcul la derivee de l'erreur: gradient du coût par rapport aux
            entrées en fonction de l'entrée input et des δ de la couche
            suivante (delta).
        """
        pass


class Linear(Module):
    """ Module représentant une couche linéaire avec input entrées et output
        sorties.
            * self._parameters: float array x array, matrice de poids pour la
              couche linéaire, de taille input x output.
    """
    def __init__(self, input, output):
        """ Constructeur du module Linear.
        """
        self.input = input
        self.output = output
        self._parameters = 2 * ( np.random.rand(self.input, self.output) - 0.5 )
        self.zero_grad()

    def zero_grad(self):
        """ Réinitialise le gradient à 0.
        """
        self._gradient = np.zeros((self.input, self.output))

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
            Elle prend en entrée une matrice de taille batch * input et rend en
            sortie une matrice de taille batch * output.
            @param X: (float) array x array, matrice des entrées (batch x input)
        """
        return np.dot( X, self._parameters)

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, X, delta):
        """ Met a jour la valeur du gradient: calcule le gradient du coût par
            rapport aux paramètres et l'additionne à la variable self._gradient
            en fonction de l'entrée input et des δ de la couche suivante (delta).
            Le gradient calculé est de taille input x output
            @param X: (float) array x array, matrice des entrées (batch x input)
            @param delta: (float) array x array, matrice de dimensions
                          batch x output
        """
        self._gradient += np.dot( X.T, delta )

    def backward_delta(self, X, delta):
        """ Calcul la derivee de l'erreur: gradient du coût par rapport aux
            entrées en fonction de l'entrée input et des δ de la couche
            suivante (delta).
        """
        return np.dot( delta, self._parameters.T )


class TanH(Module):
    """ Module représentant une couche de transformation tanh.
    """
    def __init__(self):
        """ Constructeur du module TanH.
        """
        super().__init__()

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
            Elle prend en entrée une matrice de taille batch * input et rend en
            sortie une matrice de taille batch * input.
            @param X: (float) array x array, matrice des entrées (batch x input)
        """
        return np.tanh(X)

    def backward_delta(self, X, delta):
        """ Calcul la dérivée de l'erreur du module TanH par rapport aux
            δ de la couche suivante (delta).
        """
        return ( 1 - np.tanh(X)**2 ) * delta

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        pass


class Sigmoide(Module):
    """ Module représentant une couche de transformation sigmoide.
    """
    def __init__(self):
        """ Constructeur du module Sigmoide.
        """
        super().__init__()

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
            Elle prend en entrée une matrice de taille batch * input et rend en
            sortie une matrice de taille batch * input.
            @param X: (float) array x array, matrice des entrées (batch x input)
        """
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, X, delta):
        """ Calcul la dérivée de l'erreur du module Sigmoide par rapport aux
            δ de la couche suivante (delta).
        """
        #print("SIGMOIDE =",( np.exp(-X) / ( 1 + np.exp(-X) )**2 ))
        #input()
        return ( np.exp(-X) / ( 1 + np.exp(-X) )**2 ) * delta

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        pass


class SoftMax (Module) :
    """ Module représentant une couche de transformation SoftMax.
    """
    def __init__(self):
        """ Constructeur du module SoftMax.
        """
        super().__init__()

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
            Elle prend en entrée une matrice de taille batch * input et rend en
            sortie une matrice de taille batch * input.
            @param X: (float) array x array, matrice des entrées (batch x input)
        """
        e_x = np.exp(X)
        return e_x / e_x.sum( axis = 1 ).reshape(-1,1)

    def backward_delta(self, X, delta):
        """ Calcul la dérivée de l'erreur du module Sigmoide par rapport aux
            δ de la couche suivante (delta).
        """
        s = self.forward( np.array(X) )
        return s * ( 1 - s ) * delta

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        pass


############### NOTRE TOUT PREMIER RESEAU: REGRESSION LINEAIRE ###############

class RegLin:
    """ Classe pour la régression linéaire par réseau de neurones.
    """
    def fit(self, xtrain, ytrain, niter=100, gradient_step=1e-5):
        """ Réalise la régression linéaire sur les données d'apprentissage.
            @param xtrain: float array x array, données d'apprentissage
            @param ytrain: int array, labels sur les données d'apprentissage
            @param niter: int, nombre d'itérations
            @param gradient_step: float, pas de gradient
        """
        # Ajout d'un biais aux données
        #xtrain = add_bias(xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

        # Initialisation de la couche et de la loss
        self.mse = MSELoss()
        self.linear = Linear(input, output)
        self.list_loss=[]
        for i in range(niter):

            # ETAPE 1: Calcul de l'état du réseau (phase forward)
            yhat = self.linear.forward(xtrain)

            # ETAPE 2: Phase backward (rétro-propagation du gradient de la loss
            #          par rapport aux paramètres et aux entrées)
            last_delta = self.mse.backward(ytrain, yhat)
            delta = self.linear.backward_delta(xtrain, last_delta)

            self.linear.backward_update_gradient(xtrain, delta)

            # ETAPE 3: Mise à jour des paramètres du réseau (matrice de poids w)
            self.linear.update_parameters(gradient_step)
            self.linear.zero_grad()
            self.list_loss.append(np.mean( self.mse.forward(ytrain, yhat) ))
        # Calcul de la loss
        self.last_loss = np.mean( self.mse.forward(ytrain, yhat) )

    def predict(self, xtest):
        """ Prédiction sur des données. Il s'agit simplement d'un forward sur
            la couche linéaire.
        """
        return self.linear.forward(xtest)


############ NOTRE DEUXIEME RESEAU: CLASSIFICATION NON LINEAIRE ##############

class NonLin:
    """ Classe pour un classifieur non-linéaire par réseau de neurones.
    """
    def fit(self, xtrain, ytrain, niter=100, gradient_step=1e-5, neuron=100):
        """ Classification non-linéaire sur les données d'apprentissage.
            @param xtrain: float array x array, données d'apprentissage
            @param ytrain: int array, labels sur les données d'apprentissage
            @param niter: int, nombre d'itérations
            @param gradient_step: float, pas de gradient
            @param neuron: nombre de neurones dans une couche
        """
        # Ajout d'un biais aux données
        xtrain = add_bias (xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

        # Initialisation des couches du réseau et de la loss
        self.mse = MSELoss()
        self.linear_1 = Linear(input, neuron)
        self.tanh = TanH()
        self.linear_2 = Linear(neuron, output)
        self.sigmoide = Sigmoide()

        for i in range(niter):

            # ETAPE 1: Calcul de l'état du réseau (phase forward)
            res1 = self.linear_1.forward(xtrain)
            res2 = self.tanh.forward(res1)
            res3 = self.linear_2.forward(res2)
            res4 = self.sigmoide.forward(res3)

            # ETAPE 2: Phase backward (rétro-propagation du gradient de la loss
            #          par rapport aux paramètres et aux entrées)
            last_delta = self.mse.backward(ytrain, res4)

            delta_sig = self.sigmoide.backward_delta(res3, last_delta)
            delta_lin = self.linear_2.backward_delta(res2, delta_sig)
            delta_tan = self.tanh.backward_delta(res1, delta_lin)

            self.linear_1.backward_update_gradient(xtrain, delta_tan)
            self.linear_2.backward_update_gradient(res2, delta_sig)
            # ETAPE 3: Mise à jour des paramètres du réseau (matrice de poids w)
            self.linear_1.update_parameters(gradient_step)
            self.linear_2.update_parameters(gradient_step)
            self.linear_1.zero_grad()
            self.linear_2.zero_grad()

        # Affichage de la loss
        print("\nErreur mse :", np.mean( self.mse.forward(ytrain, res4) ) )

    def predict(self, xtest):
        """ Prédiction sur des données. Il s'agit simplement d'un forward.
        """
        # Ajout d'un biais aux données
        xtest = add_bias (xtest)

        # Phase passe forward
        res = self.linear_1.forward(xtest)
        res = self.tanh.forward(res)
        res = self.linear_2.forward(res)
        res = self.sigmoide.forward(res)

        return np.argmax(res, axis = 1)


######################## CLASSE SEQUENTIEL ET OPTIM ##########################

class Sequentiel:
    """ Classe qui permet d'ajouter des modules en série et d'automatiser les
        procédure forward et backward sur toutes les couches.
            * self.modules: list(Module), liste des modules du réseau
            * self.loss: Loss, coût à minimiser
    """
    def __init__(self, modules, loss):
        """ Constructeur de la classe Sequentiel.
        """
        self.modules = modules
        self.loss = loss

    def fit(self, xtrain, ytrain):
        """ Réalise une itération forward et backward sur les couches du
            paramètre self.modules.
            @param xtrain: float array x array, données d'apprentissage
            @param ytrain: int array, labels sur les données d'apprentissage
        """
        # ETAPE 1: Calcul de l'état du réseau (phase forward)
        res_forward = [ self.modules[0].forward(xtrain) ]

        for j in range(1, len(self.modules)):
            res_forward.append( self.modules[j].forward( res_forward[-1] ) )

        # ETAPE 2: Phase backward (rétro-propagation du gradient de la loss
        #          par rapport aux paramètres et aux entrées)
        deltas =  [ self.loss.backward( ytrain, res_forward[-1] ) ]

        for j in range(len(self.modules) - 1, 0, -1):
            deltas += [ self.modules[j].backward_delta( res_forward[j-1], deltas[-1] ) ]

        return res_forward, deltas


class Optim:
    """ Classe qui permet de condenser une itération de gradient. Elle calcule
        la sortie du réseau self.net, exécute la passe backward et met à jour
        les paramètres du réseau.
            * self.net: list(Module), réseau de neurones sous forme d'une liste
                        de Modules correspondant aux différentes couches.
            * self.loss: Loss, coût à minimiser
            * self.eps: float, pas pour la mise-à-jour du gradient
    """
    def __init__(self, net, loss, eps):
        """ Constructeur de la classe Optim.
        """
        self.net = net
        self.loss = loss
        self.eps = eps
        self.sequentiel = Sequentiel(net, loss)

    def step(self, batch_x, batch_y):
        """ Calcule la sortie du réseau, exécute la passe-backward et met à
            jour les paramètres du réseau.
            @param batch_x: float array x array, batch d'apprentissage
            @param batch_y: int array, labels sur le batch d'apprentissage
        """
        # ETAPE 1: Calcul de l'état du réseau (phase forward) et passe backward
        res_forward, deltas = self.sequentiel.fit(batch_x, batch_y)

        # ETAPE 2: Phase backward par rapport aux paramètres et mise-à-jour
        for j in range(len(self.net)):

            # Mise-à-jour du gradient
            if j == 0:
                self.net[j].backward_update_gradient(batch_x, deltas[-1])
            else:
                self.net[j].backward_update_gradient(res_forward[j-1], deltas[-j-1])

            # Mise-à-jour des paramètres
            self.net[j].update_parameters(self.eps)
            self.net[j].zero_grad()

    def predict(self, xtest, onehot=False):
        """ Prédiction sur des données. Il s'agit simplement d'un forward.
        """
        # Phase passe forward
        res_forward = [ self.net[0].forward(xtest) ]

        for j in range(1, len(self.net)):
            res_forward.append( self.net[j].forward( res_forward[-1] ) )
        
        yhat = np.argmax(res_forward[-1], axis=1)
        
        if onehot:
            onehot = np.zeros((yhat.size, len(np.unique(yhat))))
            onehot[ np.arange(yhat.size), yhat ] = 1
            yhat = onehot
        
        return yhat


###### NOTRE TROISIEME RESEAU: CLASSIFICATION NON-LINEAIRE SEQUENTIELLE ######

class NonLin2:
    """ Classe pour un classifieur non-linéaire par réseau de neurones,
        version séquentielle. La fonction fit correspond à la classe SGD
        demandée.
    """
    def fit(self, xtrain, ytrain, batch_size=1, neuron=10, niter=1000, gradient_step=1e-5):
        """ Classifieur non-linéaire sur les données d'apprentissage.
            @param xtrain: float array x array, données d'apprentissage
            @param ytrain: int array, labels sur les données d'apprentissage
            @param batch_size: int, taille des batchs
            @param neuron: nombre de neurones dans une couche
            @param niter: int, nombre d'itérations
            @param gradient_step: float, pas de gradient
        """
        # Ajout d'un biais aux données
        xtrain = add_bias (xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

        # Initialisation de la listte des loss
        self.list_loss = []
        
        # Initialisation des couches du réseau et de la loss
        self.mse = MSELoss()
        self.linear_1 = Linear(input, neuron)
        self.tanh = TanH()
        self.linear_2 = Linear(neuron, output)
        self.sigmoide = Sigmoide()

        # Liste des couches du réseau de neurones
        modules = [ self.linear_1, self.tanh, self.linear_2, self.sigmoide ]

        # Apprentissage du réseau de neurones
        self.optim  = Optim(modules, self.mse, gradient_step)

        for i in range(niter):
            # Tirage d'un batch de taille batch_size et mise-à-jour
            inds = [ rd.randint(0, len(xtrain) - 1) for j in range(batch_size) ]
            self.optim.step( xtrain[inds], ytrain[inds] )
            self.list_loss.append(np.mean( self.mse.forward(ytrain, self.optim.predict(xtrain)) ))
        

    def SGD(self, net, loss, xtrain, ytrain, batch_size=1, niter=1000, gradient_step=1e-5):
        # Ajout d'un biais aux données
        xtrain = add_bias (xtrain)
        
        # Initialisation de la listte des loss
        self.list_loss = []
        
        # Apprentissage du réseau de neurones
        self.optim  = Optim(net, loss, gradient_step)
        
        # Liste de variables pour simplifier la création des batchs
        card = xtrain.shape[0]
        nb_batchs = card//batch_size
        inds = np.arange(card)

        # Création des batchs
        np.random.shuffle(inds)
        batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

        for i in range(niter):
            # On mélange de nouveau lorsqu'on a parcouru tous les batchs
            if i%nb_batchs == 0:
                np.random.shuffle(inds)
                batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

            # Mise-à-jour sur un batch
            batch = batchs[i%(nb_batchs)]
            self.optim.step(xtrain[batch], ytrain[batch])
            current_loss = np.mean(loss.forward(ytrain, self.optim.predict(xtrain, onehot=True)))
            self.list_loss.append(current_loss)
         

    def predict(self, xtest):
        """ Prédiction sur des données. Il s'agit simplement d'un forward.
        """
        # Ajout d'un biais aux données
        xtest = add_bias (xtest)

        # Phase passe forward
        return self.optim.predict(xtest)
    
    
#################### NOTRE CINQUIÈME RESEAU: AUTO-ENCODEUR ###################

class AutoEncodeur :
    """ Classe pour l'auto-encodage (réduction des dimensions, compression de 
        l'information).
    """
    def codage (self, xtrain, modules):
        """ Phase d'encodage.
        """
        res_forward = [ modules[0].forward(xtrain) ]

        for j in range(1, len(modules)):
            #print("AFFICHAGE",type(modules[j]).__name__,res_forward[-1])

            res_forward.append( modules[j].forward( res_forward[-1] ) )

        return res_forward

    def fit(self, xtrain, ytrain, batch_size=1, neuron=10, niter=1000, gradient_step=1e-5):
        """ Classifieur non-linéaire sur les données d'apprentissage.
            @param xtrain: float array x array, données d'apprentissage
            @param ytrain: int array, labels sur les données d'apprentissage
            @param batch_size: int, taille des batchs
            @param neuron: nombre de neurones dans une couche
            @param niter: int, nombre d'itérations
            @param gradient_step: float, pas de gradient
        """
        # Ajout d'un biais aux données
        #xtrain = add_bias (xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

        # Initialisation des couches du réseau et de la loss
        self.bce = BCE()
        self.linear_1 = Linear(input, neuron)
        self.tanh = TanH()
        self.linear_2 = Linear(neuron, output)
        self.sigmoide = Sigmoide()
        self.linear_3 = Linear (output, neuron)
        self.linear_4 = Linear (neuron, input)


        # Liste des couches du réseau de neurones
        self.modules_enco = [ self.linear_1, self.tanh, self.linear_2, self.tanh ]
        self.modules_deco = [ self.linear_3, self.tanh, self.linear_4, self.sigmoide ]
        self.net = self.modules_enco + self.modules_deco

        for i in range(niter):
            res_forward_enco = self.codage(xtrain, self.modules_enco)
            res_forward_deco = self.codage(res_forward_enco[-1], self.modules_deco)
            res_forward = res_forward_enco + res_forward_deco
            #print("SIG", res_forward[-1])
            #print("RES FORWARD:",res_forward)
            if(i%100==0):
                print(np.sum(np.mean(self.bce.forward(xtrain, res_forward[-1]), axis=1)))

            # Phase backward (rétro-propagation du gradient de la loss
            #          par rapport aux paramètres et aux entrées)

            deltas =  [ self.bce.backward( xtrain, res_forward[-1] ) ]

            for j in range(len(self.net) - 1, 0, -1):
                deltas += [self.net[j].backward_delta( res_forward[j-1], deltas[-1] ) ]


            #Phase backward par rapport aux paramètres et mise-à-jour
            for j in range(len(self.net)):
                # Mise-à-jour du gradient
                if j == 0:
                    self.net[j].backward_update_gradient(xtrain, deltas[-1])
                else:
                    self.net[j].backward_update_gradient(res_forward[j-1], deltas[-j-1])

                # Mise-à-jour des paramètres
                self.net[j].update_parameters(gradient_step)
                self.net[j].zero_grad()

    def predict (self, xtest) :
        """ Prédiction sur des données de test.
        """
        res_forward_enco = self.codage(xtest, self.modules_enco)
        res_forward_deco = self.codage(res_forward_enco[-1], self.modules_deco)
        return res_forward_enco[-1], res_forward_deco[-1]


##############################################################################
# ------------------------------ UTILITAIRES ------------------------------- #
##############################################################################

def add_bias(datax):
    """ Fonction permettant d'ajouter un biais aux données.
        @param xtrain: float array x array, données auxquelle ajouter un biais
    """
    bias = np.ones((len(datax), 1))
    return np.hstack((bias, datax))


######################## CHARGEMENT DES DONNEES USPS #########################

def load_usps(filename):
    """ Fonction de chargement des données.
        @param filename: str, chemin vers le fichier à lire
        @return datax: float array x array, données
        @return datay: float array, labels
    """
    with open(filename, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)


def get_usps(l, datax, datay):
    """ Fonction permettant de ne garder que 2 classes dans datax et datay.
        @param l: list(int), liste contenant les 2 classes à garder
        @param datax: float array x array, données
        @param datay: float array, labels
        @param datax_new: float array x array, données pour 2 classes
        @param datay_new: float array, labels pour 2 classes
    """
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy

    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    datax_new, datay_new = np.vstack(tmp[0]),np.hstack(tmp[1])

    return datax_new, datay_new


def show_usps(datax):
    """ Fonction d'affichage des données.
    """
    plt.imshow(datax.reshape((16,16)),interpolation="nearest",cmap="magma")


def plot(datax, datay, model, name=''):
    """ Fonction d'affichage des données gen_arti et de la frontière de
        décision.
    """
    plot_frontiere(datax,lambda x : model.predict(x),step=100)
    plot_data(datax,datay.reshape(1,-1)[0])
    plt.title(name)
    plt.show()

def load_data(classes=10):
    
    
    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    # Création des données d'apprentissage et des données d'entraînement
    xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
    xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)
    
    #bruit
    bruit = np.random.rand(xtest.shape[0], xtest.shape[1])
    #xtrain = np.where(xtrain+bruit <= 2, xtrain+bruit, xtrain)
    xtest = np.where(xtest+bruit <= 2, xtest+bruit, xtest)
    
    # Normalisation
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    scaler = StandardScaler()
    xtest = scaler.fit_transform(xtest)
    
    # One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot
    
    return xtrain, ytrain, xtest, ytest



##############################################################################
# ----------------------------- FONCTION TEST------------------------------- #
##############################################################################


def mainLineaire(a=97.28, sigma=50, niter=1000):
    # On décide arbitrairement d'un coefficient directeur a
    print('\nCoefficient directeur réel:', a)
    
    # Génération des données d'entraînement
    xtrain = np.array( [ x for x in np.linspace(0, 2.0, 100) ] ).reshape(-1,1)
    ytrain = np.array( [ a * x + rd.uniform(-sigma,sigma) for x in np.linspace(0,2.0,100) ] )
    
    # Création de notre modèle de régression linéaire
    rl = RegLin()
    
    # Phase d'entraînement puis prédiction des classes des données de xtrain
    rl.fit(xtrain, ytrain.reshape(-1,1), niter=niter, gradient_step=1e-5)
    
    w = rl.linear._parameters[0][0]
    print('Coefficient linéaire prédit:', w)
    
    # Affichage de la loss
    print("\nErreur mse :", rl.last_loss )
        
    # Affichage des données et de la droite prédite
    toPlot = [ w * x[0] for x in xtrain ]
    plt.figure()
    plt.title('Régression linéaire, a = {}, â = {}, erreur = {}'.format(a, round(rl.linear._parameters[0][0], 2), round(rl.last_loss, 1)))
    plt.scatter(xtrain.reshape(1,-1), ytrain, s = 1, c = 'midnightblue', label='data')
    plt.plot(xtrain.reshape(1,-1)[0], toPlot, color = 'mediumslateblue', label='model')
    plt.legend()

    plt.figure()
    plt.title('Evolution de la loss')
    plt.plot(rl.list_loss, label='loss', c='darkseagreen')
    plt.legend()
    plt.xlabel('Nombre d\'itérations')


def mainNonLineaire(neuron=10, niter=1000, gradient_step=1e-3, batch_size=None):    
    # Création de données artificielles suivant 4 gaussiennes
    datax, datay = gen_arti(epsilon=0.1, data_type=1)
    
    # Descente de gradient batch par défaut
    if batch_size == None:
        batch_size = len(datay)
        
    # Normalisation des données
    scaler = StandardScaler()
    datax = scaler.fit_transform(datax)
    
    # One-Hot Encoding
    datay = np.array([ 0 if d == -1 else 1 for d in datay ])
    onehot = np.zeros((datay.size, 2))
    onehot[ np.arange(datay.size), datay ] = 1
    datay = onehot
    
    # Création et test sur un réseau de neurones non linéaire
    
    time_start = time.time()
    
    batch, output = datay.shape
    batch, input = datax.shape
    
    mse = MSELoss()
    linear_1 = Linear(input+1, neuron)
    tanh = TanH()
    linear_2 = Linear(neuron, output)
    sigmoide = Sigmoide()
    
    net = [linear_1, tanh, linear_2, sigmoide]
    
    nonlin = NonLin2()
    nonlin.SGD(net, mse, datax, datay, batch_size=batch_size, niter=niter, gradient_step=gradient_step)
    #nonlin.fit(datax, datay, batch_size=len(datay), niter=1000, neuron=10, gradient_step=1e-3)
    
    # Test sur les données d'apprentissage
    ypred = nonlin.predict(datax)
    datay = np.argmax(datay, axis=1)
    
    print("\nTemps d'exécution: ", time.time() - time_start )
    print("Score de bonne classification: ", np.mean( ypred == datay ))
    plot(datax, datay, nonlin, name='Regression non Linéaire, n_neurons = {}, niter = {}, gradient_step = {}'.format(neuron, niter, gradient_step))
    
    # Evolution de la loss
    plt.figure()
    plt.title('Evolution de la loss')
    plt.plot(nonlin.list_loss, label='loss', c='darkseagreen')
    plt.legend()
    plt.xlabel('Nombre d\'itérations')


def mainMulti(neuron=10, niter=300, gradient_step=1e-3, batch_size=None):
    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    # Création des données d'apprentissage et des données d'entraînement
    classes = 10
    xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
    xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)
    
    # Normalisation
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    scaler = StandardScaler()
    xtest = scaler.fit_transform(xtest)
    
    # One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot
    
    # Initialisation batch_size
    if batch_size == None:
        batch_size = len(xtrain)
        
    # Récupération des tailles des entrées
    batch, output = ytrain.shape
    batch, input = xtrain.shape
    
    # Initialisation des couches du réseau et de la loss
    ce = CESM()
    linear_1 = Linear(input+1, neuron)
    tanh = TanH()
    linear_2 = Linear(neuron, output)
    softmax = SoftMax()
    
    # Liste des couches du réseau de neurones
    net = [linear_1, tanh, linear_2, softmax]
    
    # Création et test sur un réseau de neurones non linéaire
    
    time_start = time.time()
    
    nonlin = NonLin2()
    nonlin.SGD(net, ce, xtrain, ytrain, batch_size=batch_size, niter=niter, gradient_step=gradient_step)
    #nonlin.fit(xtrain, ytrain, batch_size=len(ytrain), niter=100, gradient_step=0.01)
    
    # Evolution de la loss
    plt.figure()
    plt.title('Evolution de la CESM loss sur {} neurones, niter = {}, gradient_step = {}'.format(neuron, niter, gradient_step))
    plt.plot(nonlin.list_loss, label='loss', c='darkseagreen')
    plt.legend()
    plt.xlabel('Nombre d\'itérations')
    
    # Test sur les données d'apprentissage
    ypred = nonlin.predict(xtest)
    
    print("\nTemps d'exécution: ", time.time() - time_start )
    print("Score de bonne classification: ", np.mean( ypred == ytest ))
    
    return ypred

def mainAutoEncodeur_():
    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    # Création des données d'apprentissage et des données d'entraînement
    classes = 10
    xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
    xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)
    
    # Normalisation
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    scaler = StandardScaler()
    xtest = scaler.fit_transform(xtest)
    
    ## One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot
    
    # Initialisation de l'auto-encodeur
    auto = AutoEncodeur()
    auto.fit(xtrain, ytrain, niter=500, neuron=100, gradient_step=1e-4)
    #ytrain = np.argmax(ytrain, axis=1)
    
    # Test sur les données d'apprentissage
    y_enco,y_deco= auto.predict(xtest)

def usps_data(classes=10, a_bruit=False):
    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    # Création des données d'apprentissage et des données d'entraînement
    xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
    xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)
    
    if a_bruit :
        #bruit
        bruit = np.random.rand(xtest.shape[0], xtest.shape[1])
        #xtrain = np.where(xtrain+bruit <= 2, xtrain+bruit, xtrain)
        xtest = np.where(xtest+bruit <= 2, xtest+bruit, xtest)
    
    # Normalisation
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    scaler = StandardScaler()
    xtest = scaler.fit_transform(xtest)
    
    # One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot
    
    return xtrain, ytrain, xtest, ytest

def mainAutoEncodeur(neuron = 100, classes=10, niter=500, gradient_step=1e-4):
    """
        Tests de la partie auto-encodeur sur les données manuscrites (usps).
    """
    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    # Création des données d'apprentissage et des données d'entraînement
    classes = 10
    xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
    xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)
    
    # Normalisation
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    scaler = StandardScaler()
    xtest = scaler.fit_transform(xtest)
    
    ## One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot
    
    time_start = time.time()
    
    # pour auto-encodeur
    auto = AutoEncodeur()
    auto.fit(xtrain, ytrain, ytrain.shape[1], niter=niter, neuron=neuron, gradient_step=gradient_step)
    
    # Test sur les données d'apprentissage
    y_enco,y_deco= auto.predict(xtest)
    
    kmeans = KMeans(n_clusters = 10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
    pred_enco = kmeans.fit_predict(y_enco)
    pred_deco = kmeans.fit_predict(y_deco)
    
    liste_test = [205,540,780,863,1027,1312,1505,1576,1741,1835]
    
    plt.figure(figsize=(15,4))

    for i in range(10):
        x = liste_test[i]
        print(x)
        plt.subplot(2,10, i+1)
        show_usps(xtest[x])
        plt.subplot(2,10, i+11)
        show_usps(y_deco[x])
        
    plt.savefig("auto_encodeur_bce.pdf")
    plt.show()  
    
    #t-sne
    perplexity_list = [5, 10, 30, 50, 100]
    tsne_perp = [ TSNE(n_components=2, random_state=0, perplexity=perp) for perp in perplexity_list]
    tsne_perp_data = [ tsne.fit_transform(y_deco) for tsne in tsne_perp]
    plt.figure(figsize=(15,15))
    for i in range(len(perplexity_list)):
        plt.subplot(3,2, i+1)
        plt.title("Perplexity " + str(perplexity_list[i]))
        plt.scatter(tsne_perp_data[i][:,0],tsne_perp_data[i][:,1], c=ytest, label=ytest)
    plt.subplot(3,2, i+2)
    plt.axis('off')
    plt.legend()
    print("\nTemps d'exécution: ", time.time() - time_start )