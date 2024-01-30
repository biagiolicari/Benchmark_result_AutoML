import os
import time

import pandas as pd
from sklearn.decomposition import PCA
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_sse import sse
from csmartml import csmartml as cm
from collections import OrderedDict

import redis


def initialize_app():
	app = Flask(__name__)
	app.config["REDIS_URL"] = "redis://localhost:6379"
	app.register_blueprint(sse, url_prefix='/stream')
	CORS(app)

	@app.route('/')
	def index():
		# return "Hello World!"
		return render_template("index.html")

	@app.route('/publish')
	def publish(data, type):
		sse.publish(data, type=type)
		# sse.publish({"message": "Hello World!"}, type='message')
		return "Message sent!"\

	@app.route('/disable', methods=["POST"])
	def disable():
		task_data = request.get_json(force=True)
		print(task_data)

	@app.route("/taskrun", methods=["POST"])
	def taskrun():
		# Get task configurations
		global file
		task_data = request.get_json(force=True)
		dataset = task_data["dataset"]
		time_budget = int(task_data["time"])
		resultPreference = task_data["result"]

		# time_budget = 10
		uploadedData = None
		POP_SIZE = 10

		# Load benchmark dataset or get uploaded dataset
		if "ud" in task_data.keys():
			td = task_data["ud"]
			data = pd.DataFrame.from_dict(td, columns=None)
			data = data.iloc[:-1]
			uploadedData = data
		else:
			file = "./csmartml/datasets/{}.csv".format(dataset)
			data = pd.read_csv(file, header=None, na_values='?')
			data = data.iloc[:, :-1]

		# Initialize smart clustering framework
		if "algorithm" in task_data.keys():
			alg = task_data["algorithm"]
			metric = task_data["metric"]
			comb = cm.CSmartML(dataset, POP_SIZE, time_budget, publish, False, algorithm=alg, cvi=metric,
							   dataset=uploadedData, result=resultPreference)
			pops, algorithm = comb.search(publish)
		else:
			comb = cm.CSmartML(dataset, POP_SIZE, time_budget, publish, True, dataset=uploadedData, result=resultPreference)
			pops, algorithm = comb.search(publish)

		while pops is None:
			time.sleep(time_budget)

		i = 0
		configurations = {}
		configuration_values = {}
		for pop in pops:
			cluster = pop[0].fit(data)
			configurations["CONFIG-{}".format(i)] = cluster.labels_.tolist()
			configuration_values["CONFIG-{}".format(i)] = str(pop[0])
			i += 1

		pca_2d = PCA(n_components=2).fit(data).transform(data)

		# configurations = OrderedDict(sorted(configurations.items()))

		return {"clusters": configurations, "pca": pca_2d.tolist(), "data": data.to_numpy().tolist(), "configs": configuration_values}

	@app.route('/modify')
	def modify():
		return jsonify({"STATUS": "Hello World!"})

	@app.route('/redis')
	def reorder():
		is_connected = redis.from_url("redis://localhost:6379")
		name = is_connected.acl_users()
		return jsonify({"Client": name})

	return app


def startapp():
	app = initialize_app()
	app.run(
		host="0.0.0.0",
		port="5555",
		debug=True
	)


if __name__ == "__main__":
	startapp()

