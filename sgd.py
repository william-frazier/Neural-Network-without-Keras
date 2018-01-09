# CS 451 HW 2
#William Frazier
# based on an assignment by Joe Redmon

from math import exp
import random


def logistic(x):
    """
    Takes a value and runs it throught the sigmoid function.
    """
    
    return 1 / (1 + exp(-x))

def dot(x, y):
    """
    Calculates the dot product of two lists.
    """
    
    s = 0
    for i in range(len(x)):
        s += x[i-1] * y[i-1]
    return s

def predict(model, point):
    """
    Calculates the dot product of the model and the fed point and runs that
    result through the sigmoid function.
    """
    
    return logistic(dot(model, point['features']))


def accuracy(data, predictions):
    """
    Gives the accuracy of a model by taking data and comparing its y values 
    to the given predictions.
    """
    
    correct = 0
    count = -1
    for i in data:
        test = 0
        #defaults to negative state, expecting a 0
        count += 1
        if predictions[count] >= 0.5:
            test = 1
            #if 0.5 or higher is predicted, change to expecting a 1
        if i['label'] == test:
            correct += 1
    return float(correct)/len(data)

def update(model, point, alpha, lambd):
    """
    Uses stochastic gradient descent to update the model.
    """
    
    hyp = predict(model, point)
    #hyp = hypothesis
    error = hyp - point['label']
    lam_mod = model[:]
    lam_mod[0] = 0
    #the two preceeding lines of code simply ensure theta_0 isn't calculated
    #in the regularization
    for i in range(len(model)):
        model[i] = model[i] - alpha * (error * point['features'][i] + lambd * lam_mod[i])
    return model #or is this pass, I'm unclear

def initialize_model(k):
    """
    This isn't my function but it appears to create a randomized model.
    """
    
    return [random.gauss(0, 1) for x in range(k)]

def train(data, epochs, alpha, lambd):
    """
    Works with the update function to build an increasingly more accurate
    model for predictions.
    """
    
    m = len(data) # number of training examples
    n = len(data[0]['features']) # number of features (+ 1)
    model = initialize_model(n)
    for i in range(epochs * m):
        #this could easily be done as two separate loops but this works
        update(model, data[random.randint(0,m-1)], alpha, lambd)
    return model
        
def extract_features(raw):
    """
    Pulls features from the excel sheet to use.
    """
    
    data = []
    for r in raw:
        features = []
        features.append(1.0)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/20)
        features.append(float(r['marital'] == 'Married-civ-spouse'))
        features.append(float(r['hr_per_week'])/40)
        #features.append(float(r['country'] == 'United-States')/5)
        #Dividing by 5 gives this category less weight which improves the accuracy
        features.append(float(r['race'] == 'Amer-Indian-Eskimo' or ['race'] == 'Asian-Pac-Islander' or ['race'] == 'Other')/-5)
        features.append(float(r['type_employer'] == 'Private')/5)
        features.append(float(r['capital_gain'])/10000)
        features.append(float(r['capital_loss'])/10000)
        features.append(float(r['sex'] == "Male")/3)
        #So these are really acclectic features but it's the best I found
        #It's also impossible to tell how much I'm overfitting my data
        point = {}
        point['features'] = features
        point['label'] = int(r['income'] == '>50K')
        data.append(point)
    return data

def submission(data):
    """
    Used to finely tune the final product.
    """
    
    #I would change lambda to ensure I don't overfit but for just 
    #the test data, this is the best setup I could find
    #Usually comes in just below 84%s
    return train(data, epochs=30, alpha=0.01, lambd=0)
