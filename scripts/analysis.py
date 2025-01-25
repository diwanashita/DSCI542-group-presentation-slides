import os
import click
import pickle
import numpy as np
import pandas as pd
import altair as alt
# Simplify working with large datasets in Altair
alt.data_transformers.enable('vegafusion')
import matplotlib.pyplot as plt
import pandera as pa
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_validate, RandomizedSearchCV
import warnings
warnings.filterwarnings(action='ignore')

@click.command()
@click.option('--data', type=str, help="Location of training data")
@click.option('--preprocessor_from', type=str, help="Location of preprocessor")
@click.option('--pipeline', type=str, help="Location where pipeline object is to be saved")
@click.option('--viz', type=str, help="Location where visualizations are to be stored")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(data,preprocessor_from,pipeline,viz,seed):
    os.makedirs(os.path.dirname(viz), exist_ok=True)
    os.makedirs(os.path.dirname(data), exist_ok=True)
    os.makedirs(os.path.dirname(pipeline), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_from), exist_ok=True)

    processor=pickle.load(open(preprocessor_from,'rb'))
    X_train=pd.read_csv(os.path.join(data,'X_train.csv'))
    X_test=pd.read_csv(os.path.join(data,'X_test.csv'))
    y_train=pd.read_csv(os.path.join(data,'y_train.csv')).iloc[:,0]
    y_test=pd.read_csv(os.path.join(data,'y_test.csv')).iloc[:,0]

    model_pipeline = Pipeline(steps=[
    ('preprocessor', processor),
    ('model', LogisticRegression(random_state=seed, max_iter=2000))
    ])
    pickle.dump(model_pipeline,open(os.path.join(pipeline,'pipeline.pickle'),'wb'))

    cv_pipe=cross_validate(model_pipeline, X_train, y_train, cv=5, return_train_score=True)
    results_df=pd.DataFrame(pd.DataFrame(cv_pipe))
    print("Mean CV :scores",results_df.mean())

    param_dist = {
    "model__C": [10**i for i in range(-5,10)]
    }
    #print("Grid size: %d" % (np.prod(list(map(len, param_dist.values())))))
    random_search = RandomizedSearchCV(model_pipeline,param_dist, n_iter=15, n_jobs=-1,return_train_score=True,random_state=seed)
    random_search.fit(X_train,y_train)

    #Display best parameter values along with mean scores
    b=pd.DataFrame(pd.DataFrame(random_search.cv_results_).iloc[4])
    print("Best parameter values along with mean scores:",b.T[["params","mean_train_score","mean_test_score"]])
    optimized_pipe= Pipeline(steps=[
    ('preprocessor', processor),
    ('model', LogisticRegression(random_state=123, max_iter=2000,C=random_search.best_params_['model__C']))
    ])
    optimized_pipe.fit(X_train,y_train)
    pickle.dump(optimized_pipe,open(os.path.join(pipeline,'best_pipeline.pickle'),'wb'))
    predictions=optimized_pipe.predict(X_test)

    # Define order explicitly to reorder bars
    order=["Short","Medium","Long"]
    fig,ax=plt.subplots(1,2)
    fig.set_dpi(150)
    fig.set_label(["Actual Delays","Predicted Delays"])
    ax[0].hist( y_test, label='Actual Delays')
    ax[0].set_xticklabels(labels=order)
    ax[0].set_ylabel('Frequency')
    ax[1].hist( predictions, label='Predicted Delays',color="orange")
    ax[1].set_xticklabels(labels=order)
    ax[1].set_ylabel('Frequency')
    fig.legend(loc="upper left",bbox_to_anchor=(0.01, 1.05))
    fig.suptitle('Comparison of Actual vs. Predicted Delays', y=1.12)

    plt.tight_layout()
    plt.savefig(os.path.join(viz,'PredictedVsActual.png'), bbox_inches='tight')


if __name__=='__main__':
    main()