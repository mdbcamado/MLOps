#from dagster import repository
#from pipeline.pipeline import fruit_classification_pipeline  # Adjust the path as needed

#@repository
#def my_repository():
#    return [fruit_classification_pipeline]


from dagster import Definitions
from pipeline.pipeline import fruit_classification_pipeline

# Define your repository with the jobs that you want to execute
defs = Definitions(jobs=[fruit_classification_pipeline])

