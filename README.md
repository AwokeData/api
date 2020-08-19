clone the frontend
git clone https://github.com/AwokeData/frontend.git

clone this repo (make sure to clone into a directory named api)
git clone https://github.com/AwokeData/awoke_api.git api

These two directories should be at the same level

Create virtual env and Install python requirements
cd api/
python3 -m venv venv		
source venv/bin/activate	#activate virtual environemtn and install requirements here
pip install -r requirements.txt


Open two separate terminal windows. one for frontend, and one for backend

cd frontend/
npm run build
npm start

#in second terminal
cd api/
venv/bin/flask run --no-debugger  #assuming you named your virtual environment venv


