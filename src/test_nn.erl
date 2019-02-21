-module(test_nn).

-export([test_activation/0, nn_monitor/0]).

nn_monitor() ->
	receive
		{init, Sender, Synapses, HiddenLayers, OutputNeurons} ->
			Sender ! {ready, self()},
			nn_monitor(Synapses, HiddenLayers, OutputNeurons)
	end,
	nn_monitor().


feed_forward(Neurons, Values) ->
	lists:zipwith(
		fun(Neuron, Value) -> 
			Neuron ! {input, self(), Value}
		end,
		Neurons, 
		Values).

feed_backward(OutputNeurons, Predictions) ->
	lists:zipwith(
		fun(OutputNeuron, Prediction) -> 
			OutputNeuron ! {backpropagate, Prediction}
		end,
		OutputNeurons, 
		Predictions).

receive_prediction(OutputNeurons) ->
	lists:map(
		fun(OutputNeuron) ->
			receive
				{output, OutputNeuron, Value} ->
					Value;
				stop_output -> 
					io:format("Stoping output loop~n", []),
					stop
			end
		end,
		OutputNeurons).

nn_monitor(Synapses, HiddenLayers, OutputNeurons) ->
	receive
		{train, Xi, Yi} ->
			feed_forward(Synapses, Xi),
			_ = receive_prediction(OutputNeurons),
			feed_backward(OutputNeurons, Yi);
		{predict, Xi} ->
			feed_forward(Synapses, Xi),
			Predictions = receive_prediction(OutputNeurons),
			io:format("Monitor got prediction: ~w~n", [Predictions])
	end,
	nn_monitor(Synapses, HiddenLayers, OutputNeurons).
	
	% lists:map(fun(X)->X ! {input, self(), 1.0} end, Synapses),
	% Prediction = 1.0,
	% lists:map(
	% 	fun(X) ->
	% 		receive
	% 			{output, X, Activation} ->
	% 				X ! {backpropagate, Prediction},
	% 				nn_monitor(Synapses, HiddenLayers, OutputNeurons);
	% 			A ->
	% 				io:format(">>> Received ~w ~n", [A])
	% 		end
	% 	end,
	% 	OutputNeurons),
	% nn_monitor(Synapses, HiddenLayers, OutputNeurons). 


test_activation() ->
	Monitor = spawn(test_nn, nn_monitor, []),
	{Synapses, HiddenLayers, OutputNeurons} = deeplearn:spawn_nn(
		{node(), 2}, 
		[{node(), [3, 3]}, {node(), [4, 4]}], 
		{node(), 1}, 
		tanh, 
		Monitor, 
		Monitor),
	Monitor ! {init, self(), Synapses, HiddenLayers, OutputNeurons},
	receive
		{ready, Monitor} ->
			Monitor ! {train, [1.0, 1.0], [1.0]}
	end.
	
	% Monitor ! {feed_forward, [1.0, 1.0]},
	% Monitor ! {feed_backward, [1.0]}.