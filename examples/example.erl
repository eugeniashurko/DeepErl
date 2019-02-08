% Example of usage
-module(example).
-export([train_titanic/1]).


main(TrainDataF, TestDataF) ->
	% Load data
	Train = utils:read_csv(TrainDataF),
	Test = utils:read_csv(TrainDataF),
	
	% Start NN cluster
	register(monitor, self())
	Nodes = [{'node1@eugenia-XPS13', [3, 3]},
			 {'node2@eugenia-XPS13', [3, 3]},
			 {'node3@eugenia-XPS13', [3, 3]}],
	InputNode, OutputNode =
		monitor:start_nn_cluster(Nodes, tanh),
	timer:sleep(1000),

	% Shuffle data-set
	% Split train data-set into train/test

	% Train data-set
	monitor:train(Train, 100),
	Prediction = monitor:predict(Test).