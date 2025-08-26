
create_env : 
	pyenv virtualenv yellowcab 
	pyenv local yellowcab

install_deps : 
	pip install -e . 

mlflow : 
	@echo "Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5002

allow_direnv : 
	direnv allow .