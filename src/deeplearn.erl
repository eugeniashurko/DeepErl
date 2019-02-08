-module(deeplearn).
-export([step/1, sigmoid/1, tanh/1, spawn_nn/5, init_neuron_loop/4]).

% Activation functions
step(X) when X > 0 -> 1;
step(X) when X == 0 -> 0.5;
step(_) -> 0.

sigmoid(X) -> 1 / (1 + math:exp(-X)).

tanh(X) -> 2 * sigmoid(2 * X) - 1.

activate(Weights, Values, Activation) ->
	Val = lists:foldl(
		fun(X, Y) -> X + Y end,
		0,
		lists:zipwith(fun(X, Y) -> X * Y end, [1.0 | Values], Weights)),
	io:format("Activazing on ~w~n", [Val]),
	case Activation of
		step -> step(Val);
		sigmoid -> sigmoid(Val);
		tanh -> tanh(Val)
	end.

% Activation loop of a neuron waiting for all input values to be received
%activation_loop(InNeurons, Weights, Values) when length(Values) =:= length(InNeurons) ->
%	mWeights
%	activate(Weights, Values);

activation_loop(InNeurons) ->
	lists:map(fun(InNeuron) ->
			receive
				{input, InNeuron, Value} -> Value
			end
		end,
		InNeurons).


neuron_loop(Monitor, InNeurons, OutNeurons, Weights, Activation) ->
	% check whether monitor node is alive
	io:format(
		"Entered ~p neuron loop [activation=~w] with in neurons ~w , out neurons  ~w...~n",
		[self(), Activation, InNeurons, OutNeurons]),
	io:format(
		"In weights ~w~n",
		[Weights]),
	case net_adm:ping(Monitor) of
		pong -> 
			ok;
		pang ->
			io:format("CANNOT CONNECT TO MONITOR: ~p!~n", [Monitor]),
			neuron_loop(Monitor, InNeurons, OutNeurons, Weights, Activation)
	end,
	receive
		{ input, InNeuron, Value } -> 
			io:format("Received input ~w from ~p!~n", [Value, InNeuron]),
			Values = activation_loop(
				lists:delete(InNeuron, InNeurons)),
			activate(Weights, Values, Activation);
		{ update, OutNeuron, Value } -> 
			io:format("Received update ~w from ~p!~n", [Value, OutNeuron])
	end,
	neuron_loop(Monitor, InNeurons, OutNeurons, Weights, Activation).



init_neuron_loop(Monitor, OutNeurons, LayerSize, Activation) -> 
	receive
		{out, InNeurons} -> 
			% set initial neuron connection weights
			Weights = 
				[0.01 | [
				rand:normal(0,1) *
				math:sqrt(2/length(InNeurons)) || _ <- lists:seq(1, length(InNeurons)) ]],
			neuron_loop(Monitor, OutNeurons, InNeurons, Weights, Activation)
	end,
	init_neuron_loop(Monitor, OutNeurons, LayerSize, Activation).


init_layer(_, _, 0, NeuronPids, _, _) ->
	NeuronPids;
init_layer(Node, LayerSize, N, NeuronPids, Activation, NextNeurons) ->
	Pid = spawn(Node, deeplearn, init_neuron_loop, [Node, NextNeurons, LayerSize, Activation]),
	init_layer(Node, LayerSize, N-1, [Pid | NeuronPids], Activation, NextNeurons).

connect_out_neurons([_]) -> ok;
connect_out_neurons([Layer, InLayer | Tail]) ->
	lists:map(fun(N) -> N ! {out, Layer} end, InLayer),
	connect_out_neurons([InLayer | Tail]).


spawn_nn_helper(_, [], _, [LastLayer | TailNeurons], OutNeurons) -> 
	lists:map(fun(N) -> N ! {out, OutNeurons} end, LastLayer),
	connect_out_neurons([OutNeurons, LastLayer | TailNeurons]),
	[_ | Neurons] = lists:reverse([LastLayer | TailNeurons]),
	Neurons;	
spawn_nn_helper(Node, [H | TailLayers], Activation, [NextNeurons], OutNeurons) ->
	NeuronPids = init_layer(Node, H, H, [], Activation, [NextNeurons]),
	spawn_nn_helper(Node, TailLayers, Activation, [NeuronPids, NextNeurons], OutNeurons);
spawn_nn_helper(Node, [H | TailLayers], Activation, [NextNeurons | PrevNeurons], OutNeurons) ->
	NeuronPids = init_layer(Node, H, H, [], Activation, NextNeurons),
	spawn_nn_helper(Node, TailLayers, Activation,
					[NeuronPids, NextNeurons | PrevNeurons ],
					OutNeurons).


spawn_nn(Node, Layers, Activation, InputNeurons, OutNeurons) ->
	io:format("Spawning a neural network on ~p with layers ~w....~n", [Node, Layers]),
	spawn_nn_helper(Node, Layers, Activation, [InputNeurons], OutNeurons).

