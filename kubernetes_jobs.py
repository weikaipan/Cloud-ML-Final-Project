from os import path
import yaml
from kubernetes import client, config

from celery import Celery
from app import app
# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@celery.task(bind=True)
def main(self):
    print("in calling kubernetes service")
    # config.load_kube_config()
    print(path.join(path.dirname(__file__)))

    with open(path.join(path.dirname(__file__), "deploy/kubernetes/flaskapp/ml-deployment.yaml")) as f:
        dep = yaml.safe_load(f)
        k8s_beta = client.ExtensionsV1beta1Api()
        resp = k8s_beta.create_namespaced_deployment(
            body=dep, namespace="default")
        print("Deployment created. status='%s'" % str(resp.status))

if __name__ == '__main__':
    main()
