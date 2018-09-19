# animl

A python machine learning library for scikit-learn decision tree visualization and model interpretation. 

Decision trees are the fundamental building block of gradient boosting machines and Random Forests™, probably the two most popular machine learning models for structured data. Visualizing decision trees is a tremendous aid when learning these models and later, in practice, when interpreting models. Unfortunately, current visualization packages are rudimentary and not immediately helpful to the novice. For example, we couldn't find a library that could visualize how decision nodes split up feature space. Our library could be the first. It also appears uncommon for libraries to support visualizing a specific feature vector as it weaves down through a tree's decision nodes, as we could only find one image showing this (but we didn't exhaustively look through library APIs).

The visualizations are inspired by an educational animiation by [R2D3](http://www.r2d3.us/); [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/). With animl, you can visualize how the feature space is split up at decision nodes, how the training samples get ditributed in leaf nodes and how the tree makes prediction for a single observation. These operations are very critical to grasp for someone who wants to understand how classfication or regression decision tree works. 

There are lots of intricaties in the plots that Terence had to obsess over during the contruction of the library. See [How to visualize decision tree models](https://www.google.com/) for deeper discussion of our decision tree visualization tool and how we have chosen to visualize the feature-target space of a decision tree. 


## Requirements

Needs graphviz/dot lib.

```bash
pip install graphviz
```

And, on latest mac versions, you need:

```bash
brew install graphviz --with-librsvg --with-app --with-pango
```


## Usage


`dtree`: Main function to create decision tree visualization. Given a decision tree regressor or classifier, creates and returns a tree visualization using the graphviz (DOT) language.

* **Required libraries**:  
Basic libraries and imports that will (might) be needed to generate the sample visualizations shown in examples below. 
 
```bash
from sklearn.datasets import *
from sklearn import tree
from animl.viz.trees import dtreeviz
from animl.trees import *
import graphviz
```

* **Regression decision tree**:   
The default orientation of tree is top down but you can change it to left to right using `orientation="LR"`. `view()` gives a pop up window with rendered graphviz object. 

```bash
regr = tree.DecisionTreeRegressor(max_depth=2)
boston = load_boston()
regr.fit(boston.data, boston.target)

viz = dtreeviz(regr,
               boston.data,
               boston.target,
               target_name='price',
               feature_names=boston.feature_names)
              
viz.view()              
```
  
<img src=testing/samples/boston-TD-2.png width=60% height=40%>
  
  
* **Classification decision tree**:  
An additional argument of `class_names` giving a mapping of class value with class name is required for classification trees. 

```bash
classifier = tree.DecisionTreeClassifier(max_depth=2)  # limit depth of tree
iris = load_iris()
classifier.fit(iris.data, iris.target)

viz = dtreeviz(classifier, 
               iris.data, 
               iris.target,
               target_name='variety',
              feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"]  # need class_names for classifier
              )  
              
viz.view() 
```

<img src=testing/samples/iris-TD-2.png width=60% height=40% align="center">

* **Prediction path**:  
Highlights the decision nodes in which the feature value of single observation passed in argument `X` falls. Gives feature values of the observation and highlights features which are used by tree to traverse path. 
  
```bash
regr = tree.DecisionTreeRegressor(max_depth=2)  # limit depth of tree
diabetes = load_diabetes()
regr.fit(diabetes.data, diabetes.target)
X = diabetes.data[np.random.randint(0, len(diabetes.data)),:]  # random sample from training

viz = dtreeviz(regr,
               diabetes.data, 
               diabetes.target, 
               target_name='value', 
               orientation ='LR',  # left-right orientation
               feature_names=diabetes.feature_names,
               X=X)  # need to give single observation for prediction
              
viz.view()  
```
<img src=testing/samples/diabetes-LR-2-X.png width=80% height=50%>
  
* **Decision tree without scatterplot or histograms for decision nodes**:  
Simple tree without histograms or scatterplots for decision nodes. 
Use argument `fancy=False`  
  
```bash
classifier = tree.DecisionTreeClassifier(max_depth=4)  # limit depth of tree
cancer = load_breast_cancer()
classifier.fit(cancer.data, cancer.target)

viz = dtreeviz(classifier,
              cancer.data,
              cancer.target,
              target_name='cancer',
              feature_names=cancer.feature_names, 
              class_names=["malignant", "benign"],
              fancy=False )  # fance=False to remove histograms/scatterpots from decision nodes
              
viz.view() 
```

<img src=testing/samples/breast_cancer-TD-4-simple.png width=80% height=60%>


For more examples and different implementations, please see the jupyter notebook full of examples.

## Implementation guidelines

On latest MAc, make sure to install using 

```bash
brew install graphviz --with-librsvg --with-app --with-pango
```

Use `setup.py` to make sure the library gets installed properly
 
```bash 
python setup.py install -f
```


