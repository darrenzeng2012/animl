import graphviz
from animl.trees import *
from gen_samples import *

def viz_iris(orientation="TD", max_depth=3, random_state=666, fancy=True):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    iris = load_iris()

    data = pd.DataFrame(iris.data)
    data.columns = iris.feature_names

    clf = clf.fit(data, iris.target)

    # for i in range(len(iris.data)):
    for i in [60]:
        x = data.iloc[i]
        pred = clf.predict([x.values])

        shadow_tree = ShadowDecTree(clf, iris.data, iris.target,
                                    feature_names=iris.feature_names, class_names=["setosa", "versicolor", "virginica"])

        pred2 = shadow_tree.predict(x.values)
        print(f'{x} -> {pred[0]} vs mine {pred2[0]}, path = {[f"node{p.feature_name()}" for p in pred2[1]]}')
        path = [n.id for n in pred2[1]]
        if pred[0]!=pred2[0]:
            print("MISMATCH!")

    st = dtreeviz(clf, iris.data, iris.target, target_name='variety',
                  feature_names=data.columns, orientation=orientation,
                  class_names=["setosa", "versicolor", "virginica"],  # 0,1,2 targets
                  fancy=fancy,
                  highlight_path =path)

    return st

#st = viz_boston(fancy=False, orientation='TD')
st = viz_iris(fancy=True, orientation='TD')
#st = viz_digits(fancy=True, orientation='TD')
g = graphviz.Source(st)

with open("/tmp/t3.dot", "w") as f:
    f.write(st+"\n")

g.view()
