-module(deeplearn).
-export([synapse_loop/2, sigmoid/1, sigmoid_prime/1,
		 tanh/1, tanh_prime/1, spawn_nn/6, init_neuron_loop/4, neuron_loop/5]).


% Activation functions
sigmoid(X) -> 1 / (1 + math:exp(-X)).
sigmoid_prime(X) -> sigmoid(X) * (1 - sigmoid(X)).

tanh(X) -> 2 * sigmoid(2 * X) - 1.
tanh_prime(X) -> 1 - tanh(X) * tanh(X).

activate(Value, sigmoid) -> sigmoid(Value);
activate(Value, tanh) -> tanh(Value).

derivative(Value, sigmoid) -> sigmoid_prime(Value);
derivative(Value, tanh) -> tanh_prime(Value).


fold_with_weights(Values, Weights) ->
	lists:foldl(
		fun(X, Y) -> X + Y end, 0,
		lists:zipwith(
			fun(X, Y) -> X * Y end, [1.0 | Values], Weights)).

% Activation loop of a neuron waiting for all input values to be received
activation_loop(InNeuron, Value, InNeurons) ->
	lists:map(
		fun(OtherInNeuron) ->
			case OtherInNeuron of 
				InNeuron -> Value;
				_ ->
					receive
						{input, OtherInNeuron, OtherValue} ->
							io:format("[~p] Received input ~w from ~p!~n", [self(), OtherValue, OtherInNeuron]),
							OtherValue;
						stop_input -> 
							io:format("Stoping activation loop~n", []),
							stop
						% X ->
						% 	io:format("!![~p] RECEIVED WIERD ~w, waiting from: ~p ~n", [self(), X, OtherInNeuron])
					% after
					% 	2500 ->
					% 		io:format("[~p] noone sent anything, mailbox: ~w~n", [self(), c:flush()])
					end
			end
		end,
		InNeurons).

% Error loop of a neuron waiting for all error values to be received
error_loop(OutNeuron, Error, OutNeurons) ->
	lists:map(
		fun(OtherOutNeuron) ->
			case OtherOutNeuron of 
				OutNeuron -> Error;
				_ ->
					receive
						{error, OtherOutNeuron, OtherError} ->
							io:format("[~p] Received error ~w from ~p!~n", [self(), OtherError, OtherOutNeuron]),
							OtherError;
						stop_input -> 
							io:format("Stoping error estimation loop~n", []),
							stop
					end
			end
		end,
		OutNeurons).

% Loop of an input neuron sending its value to out neurons
% when receives the message {input, Value}
synapse_loop(InputMonitor, OutNeurons) ->
	receive
		{out, NewOutNeurons} ->
			io:format("[~p] New synapse output ~w ~n", [self(), NewOutNeurons]),
			synapse_loop(InputMonitor, NewOutNeurons);
		{input, InputMonitor, Value} ->
			io:format("[~p] New synapse input ~w ~n", [self(), Value]),
			lists:map(
				fun(X) -> X ! {input, self(), Value} end,
				OutNeurons),
			synapse_loop(InputMonitor, OutNeurons);
		stop_input ->
			io:format("Force activation stop ~n", []),
			lists:map(
				fun(X) -> X ! stop_input end,
				OutNeurons),
			synapse_loop(InputMonitor, OutNeurons);
		stop -> ok
	end.

% Loop of an output neuron sending its value to out neurons
% when receives the message {input, Value}
output_loop(InNeurons, OutputMonitor,
			Weights, Activation, InputValues) ->
	% check whether monitor node is alive
	% io:format("[~p] Entered output neuron loop~n", [self()]),
	receive
		{input, InNeuron, Value} -> 
			io:format("[~p] Received input ~w from ~p!~n", [self(), Value, InNeuron]),
			NewInputValues = activation_loop(InNeuron, Value, InNeurons),
			case lists:member(stop, NewInputValues) of
				false -> 
					WeightedInput = fold_with_weights(NewInputValues, Weights),
					ActivationValue = activate(WeightedInput, Activation),
					io:format("[~p] Activated successfully with ~p!~n", [self(), ActivationValue]),
					OutputMonitor ! {output, self(), ActivationValue},
					output_loop(InNeurons, OutputMonitor, Weights, Activation, NewInputValues);
				_ -> 
					io:format("Activation of ~p failed!~n", [self()]),
					not_activated
			end,
			output_loop(InNeurons, OutputMonitor, Weights, Activation, InputValues);
		{backpropagate, Prediction} ->
			WeightedInput = fold_with_weights(InputValues, Weights),
			Error = (activate(WeightedInput, Activation) - Prediction) * derivative(WeightedInput, Activation),
			io:format("->> Error ~w~n", [Error]),
			lists:zipwith(
				fun(InNeuron, Weight) -> 
					InNeuron ! {error, self(), Error * Weight}
				end, 
				InNeurons, tl(Weights)),
			output_loop(InNeurons, OutputMonitor, Weights, Activation, InputValues)
	end,
	output_loop(InNeurons, OutputMonitor, Weights, Activation, InputValues).


neuron_loop(InNeurons, OutNeurons, Weights, Activation, InputValues) ->
	% io:format("[~p] Entered hidden neuron loop, out neurons ~w~n", [self(), OutNeurons]),
	receive
		{input, InNeuron, Value} -> 
			io:format("[~p] Received input ~w from ~p!~n", [self(), Value, InNeuron]),
			NewInputValues = activation_loop(InNeuron, Value, InNeurons),
			case lists:member(stop, NewInputValues) of
				false -> 
					WeightedInput = fold_with_weights(NewInputValues, Weights),
					ActivationValue = activate(WeightedInput, Activation),
					io:format("[~p] Activated successfully with ~p!~n", [self(), ActivationValue]),
					lists:map(
						fun(X) -> X ! {input, self(), ActivationValue} end,
						OutNeurons),
					neuron_loop(InNeurons, OutNeurons, Weights, Activation, NewInputValues);
				_ -> 
					io:format("Activation of ~p failed!~n", [self()]),
					not_activated
			end;
		{error, OutNeuron, Error} ->
			io:format("[~p] Received error ~w from ~p!~n", [self(), Error, OutNeuron]),
			Errors = error_loop(OutNeuron, Error, OutNeurons),
			case lists:member(stop, Errors) of
				false -> 
					WeightedInput = fold_with_weights(InputValues, Weights),
					MyError = lists:sum(Errors) * derivative(WeightedInput, Activation),
					io:format("->> Error ~w~n", [MyError]),
					NewWeights = lists:zipwith(
						fun(InputValue, Weight) ->
							Weight - 0.02 * InputValue * MyError
						end,
						[1.0 | InputValues], Weights),
					io:format("->> Updated weights to  ~w~n", [NewWeights]),
					lists:zipwith(
						fun(InNeuron, Weight) ->
							InNeuron ! {error, self(), MyError * Weight}
						end,
						InNeurons, tl(Weights)),
					neuron_loop(InNeurons, OutNeurons, NewWeights, Activation, InputValues);
				_ -> 
					io:format("Error estimation of ~p failed!~n", [self()]),
					not_activated
			end;
		{ update, OutNeuron, Value } -> 
			io:format("Received update ~w from ~p!~n", [Value, OutNeuron])
	end,
	neuron_loop(InNeurons, OutNeurons, Weights, Activation, InputValues).


init_neuron_loop([InputMonitor], _, _, input) -> 
	receive
		{out, OutNeurons} -> 
			synapse_loop(InputMonitor, OutNeurons)
	end;
init_neuron_loop(InNeurons, [OutputMonitor], Activation, output) ->
	Weights = 
		[0.01 | [
		rand:normal(0,1) *
		math:sqrt(2/length(InNeurons)) || _ <- lists:seq(1, length(InNeurons)) ]],
	output_loop(InNeurons, OutputMonitor, Weights, Activation, 0); 
init_neuron_loop(InNeurons, _, Activation, hidden) -> 
	receive
		{out, OutNeurons} -> 
			Weights = 
				[0.01 | [
				rand:normal(0,1) *
				math:sqrt(2/length(InNeurons)) || _ <- lists:seq(1, length(InNeurons)) ]],
			neuron_loop(InNeurons, OutNeurons, Weights, Activation, 0)
	end.

init_layer(_, _, 0, NeuronPids, _, _, _, _) ->
	lists:reverse(NeuronPids);
init_layer(Node, LayerSize, N, NeuronPids, Activation, PrevNeurons, OutNeurons, NeuronType) ->
	Pid = spawn(Node, deeplearn, init_neuron_loop, [PrevNeurons, OutNeurons, Activation, NeuronType]),
	init_layer(Node, LayerSize, N-1, [Pid | NeuronPids], Activation, PrevNeurons, OutNeurons, NeuronType).

connect_out_neurons([_]) -> ok;
connect_out_neurons([[], _ | _]) -> ok;
connect_out_neurons([Layer, InLayer | Tail]) ->
	lists:map(fun(N) -> N ! {out, Layer} end, InLayer),
	connect_out_neurons([InLayer | Tail]).


spawn_layers(_, [], _, _, OutNeurons, [LastLayer | PrevNeurons], _) ->
	lists:map(fun(N) -> N ! {in, OutNeurons} end, LastLayer),
	connect_out_neurons([LastLayer | PrevNeurons]),
	lists:reverse([LastLayer | PrevNeurons]);
spawn_layers(Node, [H | TailLayers], Activation, InputNeurons, OutNeurons, [], NeuronType) ->
	NeuronPids = init_layer(
		Node, H, H, [], Activation, InputNeurons, OutNeurons, NeuronType),
	lists:map(fun(N) -> N ! {out, NeuronPids} end, InputNeurons),
	spawn_layers(
		Node, TailLayers, Activation,
		InputNeurons, OutNeurons, 
		[NeuronPids], NeuronType);
spawn_layers(Node, [H | TailLayers], Activation, InputNeurons, OutNeurons,
			    [PrevNeurons | NeuronsBefore], NeuronType) ->
	NeuronPids = init_layer(Node, H, H, [], Activation, PrevNeurons, OutNeurons, NeuronType),
	spawn_layers(Node, TailLayers, Activation,
					InputNeurons, OutNeurons,
					[NeuronPids, PrevNeurons | NeuronsBefore], NeuronType).

spawn_layers(Node, Layers, Activation, InputNeurons, OutNeurons, NeuronType) ->
	spawn_layers(
		Node, Layers, Activation,
		InputNeurons, OutNeurons, [], NeuronType).

spawn_nn_helper(Synapses, 
		 [],
		 {OutputNode, NOutputs}, Activation,
		 [LastHiddenBatch | TailNeurons], OutputMinitor) ->
	% Called for init of the last hidden layer batch, input neurons
	% are initialized to the last layer of the previous batch.
	% Output layer is initialized, output neurons of the last hidden batch
	% are initialized to output layer 
	% Initialize hidden layers
	io:format("Initializing output layer...~n", []), 
	[OutputNeurons] = spawn_layers(
		OutputNode, [NOutputs], Activation,
		lists:last(LastHiddenBatch), [OutputMinitor], output),
	% lists:map(fun(X) -> X ! {out, OutputNeurons} end, lists:last(LastHiddenBatch)),
	{Synapses, lists:reverse([LastHiddenBatch | TailNeurons]), OutputNeurons};
spawn_nn_helper(Synapses,
		 [{HeadNode, HeadLayers} | Tail],
		 Output, Activation, [], OutputMinitor) ->
	% Called for init of the first hidden layer batch, input neurons
	% are initialized to synapses
	Neurons = spawn_layers(
		HeadNode, HeadLayers, Activation,
		Synapses, [], hidden),
	% io:format("~n!!!Sending ~w, to ~w~n", [hd(Neurons), lists:last(LastHiddenBatch)]),
	lists:map(fun(X) -> X ! {out, hd(Neurons)} end, Synapses),
	spawn_nn_helper(
		Synapses, Tail, 
		Output, Activation, [Neurons], OutputMinitor);
spawn_nn_helper(Synapses,
	     [{HeadNode, HeadLayers} | Tail],
		 Output, Activation, [LastHiddenBatch | TailNeurons], OutputMinitor) ->
	% Called for init of the i-th (1 < i < n) hidden layer batch,
	% input neurons are initialized to the last layer of the previous batch
	Neurons = spawn_layers(
		HeadNode, HeadLayers, Activation,
		lists:last(LastHiddenBatch), [], hidden),
	% lists:map(fun(X) -> X ! {out, hd(Neurons)} end, lists:last(LastHiddenBatch)),
	spawn_nn_helper(
			Synapses, Tail,
		    Output, Activation,
		    [Neurons, LastHiddenBatch | TailNeurons], OutputMinitor).
	

spawn_nn(
	{InputNode, NInputs}, Hidden, Output, Activation,
	InputMonitor, OutputMinitor) ->
	% Initialize input neurons - synapses
	io:format("Initializing synapses...~n", []),
	[Synapses] = spawn_layers(
		InputNode, [NInputs], Activation, [InputMonitor], [], input),
	io:format("Initializing hidden layers...~n", []), 
	spawn_nn_helper(Synapses, Hidden, Output, Activation, [], OutputMinitor).


% network_monitor_loop(Synapses, OutNeuron) ->
% 	receive
% 		{update, synapses, NewSynapses} ->
% 			network_monitor_loop(NewSynapses, OutNeuron);
% 		{update, out_neuron, NewOutNeuron} ->
% 			network_monitor_loop(Synapses, OutNeuron);
% 		{train, DataSet}
% 	end.