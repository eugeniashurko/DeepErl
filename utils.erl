% Collection of utils for reading/writing csv files
% TODO: make distributed ?
-module(utils).
-export([read_csv/1]).

% Read csv into a dataframe with floats
read_csv(File) ->
	OpenRes = file:open(File, [read, raw]),
	case OpenRes of
		{ok, F} -> 
			parse_csv(F, file:read_line(F), []);
		_ -> io:format("Cannot open file '~s'~n", [File])
	end.

parse_csv(F, eof, Result) ->
	file:close(F),
	array:from_list(lists:reverse(Result));    
parse_csv(F, {ok, Line}, Result) ->
	parse_csv(F, file:read_line(F), [parse_line(Line)|Result]).


parse_line(Line) -> parse_line(Line, []).

parse_line([], Fields) ->
	array:from_list(lists:map(
		fun(X) -> 
			{FloatEl, _} = string:to_float(X),
			FloatEl
		end,
		lists:reverse(Fields)));
parse_line("," ++ Line, Fields) -> parse_field(Line, Fields);
parse_line(Line, Fields) ->
	parse_field(Line, Fields).

parse_field("\"" ++ Line, Fields) -> parse_field_q(Line, [], Fields);
parse_field(Line, Fields) -> parse_field(Line, [], Fields).

parse_field("," ++ _ = Line, Buf, Fields) -> parse_line(Line, [lists:reverse(Buf)|Fields]);
parse_field([C|Line], Buf, Fields) -> parse_field(Line, [C|Buf], Fields);
parse_field([], Buf, Fields) -> parse_line([], [lists:reverse(Buf)|Fields]).

parse_field_q(Line, Fields) -> parse_field_q(Line, [], Fields).
parse_field_q("\"\"" ++ Line, Buf, Fields) -> parse_field_q(Line, [$"|Buf], Fields);
parse_field_q("\"" ++ Line, Buf, Fields) -> parse_line(Line, [lists:reverse(Buf)|Fields]);
parse_field_q([C|Line], Buf, Fields) -> parse_field_q(Line, [C|Buf], Fields).