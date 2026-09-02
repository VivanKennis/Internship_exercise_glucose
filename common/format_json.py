import json
import sys

import numpy

INDENT = 4
SPACE = " "
NEWLINE = "\n"


def format_float(value: float) -> str:
    # Use Python's native repr for the shortest exact round-trip representation
    return repr(value)


def to_json(o: object, level: int = 0) -> str:
    # initialize the string that builds up the file contents
    ret = ""

    # if o is a dict - format accordingly
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        # loop over the contents of the dict
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            # recursive call to format the content
            ret += to_json(v, level + 1)

        # end of dict
        ret += NEWLINE + SPACE * INDENT * level + "}"

    # if the element is -inf - replace with string "-Infinity"
    elif o == float('-inf'):
        ret += '-Infinity'

    # if the element is inf - replace with string "Infinity"
    elif o == float('inf'):
        ret += 'Infinity'

    # if the element is nan - replace with string "NaN"
    elif isinstance(o, float) and numpy.isnan(o):
        ret += 'NaN'

    # if the element is a string - add "" marks around
    elif isinstance(o, str):
        ret += '"' + o + '"'

    # if the element is a list of lists
    elif isinstance(o, list) and all(isinstance(item, list) for item in o) and o != []:
        ret += '[\n' + SPACE * INDENT * (level + 1)

        # iterate over the lists in the list
        for oo in o:
            # recursive call to format the sub-lists
            ret += to_json(oo, level + 1) + ',\n'
            ret += SPACE * INDENT * (level + 1)

        # close the main list
        ret = ret[:-INDENT*(level + 1)-2] + '\n' + SPACE * \
            INDENT * (level + 1) + ']'
        return ret

    # if the element is a list
    elif isinstance(o, list):
        ret += '[' + ', '.join([to_json(item, level + 1) for item in o]) + ']'
        return ret

    # Tuples are interpreted as lists
    elif isinstance(o, tuple):
        ret += "[" + ", ".join(to_json(e, level + 1) for e in o) + "]"

    # if element is a boolean
    elif isinstance(o, bool):
        ret += "true" if o else "false"

    # if the element is an integer - add the value to the string
    elif isinstance(o, int):
        ret += str(o)

    # if the element is an float - call sub-function to format all float variants
    elif isinstance(o, float):
        ret += format_float(o)

    # if the element is an numpy.array containing integers
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.integer):
        ret += "[" + ', '.join(map(str, o.flatten().tolist())) + "]"

    # if the element is an numpy.array containing floats
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.inexact):
        ret += "[" + ', '.join(map(lambda x: '%.7g' %
                               x, o.flatten().tolist())) + "]"

    # if the element is None- return 'null
    elif o is None:
        ret += 'null'

    # if the element is non of the above types - throw an error
    else:
        raise TypeError(
            "Unknown type '%s' for json serialization" % str(type(o)))

    # return string with the formatted element
    return ret


def format_json_file(input_file: str, output_file: str) -> None:
    # Load the original JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

        # format the contents
        json_string = to_json(data)

        # Save the processed data back to a file
        with open(output_file, 'w') as f_out:
            f_out.write(json_string)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file_path = sys.argv[1]
    else:
        sys.exit("Error: A file to format needs to be provided.")

    if len(sys.argv) > 2:
        output_file_path = sys.argv[2]
    else:
        output_file_path = sys.argv[1]

    # format the .json file
    format_json_file(input_file_path, output_file_path)
