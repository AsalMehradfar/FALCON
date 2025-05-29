import networkx as nx
import re
from sympy import sympify
import os

from utils.io_tools import load_values, load_netlist, save_graph, load_yaml
from data_modules.graph_convertor import networkx_to_pyg


GND_LIST = load_yaml("config/data_config.yaml")["Ground"]


def preprocess_netlist(netlist_content):
    """
    Preprocess the netlist to combine multi-line entries into single lines.

    Args:
        netlist_content (list of str): List of netlist lines.

    Returns:
        list of str: Preprocessed netlist with multi-line entries combined.
    """
    combined_lines = []
    buffer = ""      

    for line in netlist_content:

        if line.startswith('//') or line.strip() == '':
            continue

        # Strip leading and trailing whitespace
        line = line.strip()
        
        # Check if the line ends with a backslash (\), indicating continuation
        if line.endswith("\\"):
            # Add the current line to the buffer, removing the backslash
            buffer += line[:-1] + ""  # Remove '\' and add a space
        else:
            # Add the final part of the line to the buffer
            buffer += line
            # Store the combined line and reset the buffer
            combined_lines.append(buffer.strip())
            buffer = ""

    return combined_lines


def parse_netlist_line(line):
    """
    Parses a single netlist line into its components.

    Args:
        line (str): A single line from the netlist.

    Returns:
        tuple: A tuple containing:
            - comp_name (str): The name of the component.
            - nodes (list of str): The list of nodes connected by the component.
            - comp_type (str): The type of the component (e.g., 'nmos', 'capacitor').
            - attributes (dict): A dictionary of the component's attributes.
        If the line doesn't match the expected format, returns (None, None, None, None).
    """
    # Use a regular expression to extract the component name, nodes, type, and attributes
    match = re.match(r'(\S+)\s+\((.*?)\)\s+(\S+)(.*)', line)
    
    if match:
        # Group 1: Component name (e.g., "N2", "C1", "R1")
        comp_name = match.group(1)
        
        # Group 2: Nodes (e.g., "net1 net2 GND") - split into a list
        nodes = match.group(2).split()
        
        # Group 3: Component type (e.g., "nmos", "capacitor")
        comp_type = match.group(3)
        
        # Group 4: Attributes (e.g., "w=W l=45.0n ...") - extract into a dictionary
        attributes = extract_attributes(match.group(4).strip())
        
        # Return the parsed components
        return comp_name, nodes, comp_type, attributes
    
    # If the line doesn't match the expected format, return None for all parts
    return None, None, None, None



def extract_attributes(attribute_string):
    """
    Extracts key-value pairs or standalone attributes from a netlist string.

    Args:
        attribute_string (str): The string containing attributes.

    Returns:
        dict: A dictionary of attributes with keys and their corresponding values.
    """
    # Regular expression to match key-value pairs
    pattern = r'(\w+)=([^=]+?)(?=\s\w+=|$)'  # Matches key=value pairs
    
    # Find all matches using regex
    matches = re.findall(pattern, attribute_string)
    
    # Convert matches to a dictionary
    attributes = {key: value.strip() for key, value in matches}
    
    return attributes


def categorize_attributes(attributes):
    """
    Categorizes attributes into three categories:
    1. Numeric values (pure numbers or those with units).
    2. Parametric values.
    3. Computed values (equations or complex expressions).

    Args:
        attributes (dict): The dictionary of attributes to categorize.

    Returns:
        tuple: Three dictionaries for numeric, parametric, and computed values.
    """
    numeric_dict = {}
    parametric_dict = {}
    computed_dict = {}

    # Regex to match numeric values with units (e.g., '45.0n', '1.23u')
    numeric_pattern = r'^\d+(\.\d+)?[a-zA-Z]*$'

    for key, value in attributes.items():
        # Handle numeric types directly
        if isinstance(value, (int, float)):
            numeric_dict[key] = value

        # Handle string types
        elif isinstance(value, str):
            value = value.strip()  # Remove any extra spaces

            # Special case: if key is 'region', always treat it as numeric
            if key == "region" or key == "type" or key == "fundname":
                numeric_dict[key] = value
            # Check for numeric values (with or without units)
            elif re.match(numeric_pattern, value):
                numeric_dict[key] = value
            # Check for parametric values (single word, no spaces or operators)
            elif re.match(r'^[a-zA-Z_]\w*$', value):
                parametric_dict[key] = value
            # Otherwise, classify as a computed value (contains operators or parentheses)
            else:
                computed_dict[key] = value

        # If the value is not a string or numeric type, classify it as computed
        else:
            computed_dict[key] = value

    return numeric_dict, parametric_dict, computed_dict


def replace_values_and_evaluate(main_dict, replacement_dict, precision=10):
    """
    Replaces values in `main_dict` with corresponding values from `replacement_dict`.
    Evaluates mathematical expressions if possible, excluding those with parametric variables.
    Converts numeric strings into numeric types and rounds numeric results.

    Args:
        main_dict (dict): The dictionary whose values need to be replaced.
        replacement_dict (dict): The dictionary containing replacement values.
        precision (int): The number of decimal places to round numeric results to.

    Returns:
        dict: The updated dictionary with replaced, evaluated, and correctly formatted values.
    """
    # Unit prefixes and their multipliers
    unit_multipliers = {
        'f': 1e-15,  # femto
        'p': 1e-12,  # pico
        'n': 1e-9,   # nano
        'u': 1e-6,   # micro
        'm': 1e-3,   # milli
        'k': 1e3,    # kilo
        'M': 1e6,    # mega
        'G': 1e9     # giga
    }

    def evaluate_expression(expression, replacements):
        """
        Replaces variables in the expression with their corresponding values
        from the replacements dictionary and evaluates the expression
        if it contains no parametric variables.

        Args:
            expression (str): The string expression to evaluate.
            replacements (dict): A dictionary of replacements for variables.

        Returns:
            str or float: The evaluated result as a number or the updated expression.
        """
        # Replace all variables in the expression
        def replacer(match):
            variable = match.group(0)
            # Skip replacement if variable starts and ends with "
            if variable.startswith('"') and variable.endswith('"'):
                return variable[1:-1]  # remove the first and last character (the quotes)
            return str(replacements.get(variable, variable))  # Replace if found, else keep as-is

        # Regex to find variables (words that are not numbers or operators)
        # pattern = r'\b[a-zA-Z_]\w*\b'
        pattern = r'"[a-zA-Z_]\w*"|\b[a-zA-Z_]\w*\b'
        updated_expression = re.sub(pattern, replacer, expression)

        # Check if the updated expression contains parametric variables
        if re.search(pattern, updated_expression):
            # Contains unresolved parametric variables; return as-is
            return updated_expression

        # Convert units to their numeric equivalents
        updated_expression = convert_units(updated_expression)

        try:
            # Evaluate the mathematical expression using sympy
            evaluated = sympify(updated_expression)
            return round(float(evaluated), precision)  # Round the numeric result to the specified precision
        except Exception as e:
            # If evaluation fails, return the updated expression
            return updated_expression

    def convert_units(expression):
        """
        Converts values with units in an expression to their numeric equivalents.

        Args:
            expression (str): The string expression to process.

        Returns:
            str: The expression with units converted to numeric equivalents.
        """
        # Regex to find numbers with units (e.g., 1.5u, 45.0n)
        unit_pattern = r'(\d+(\.\d+)?)([pnumkMG]?)'

        def unit_replacer(match):
            value = float(match.group(1))  # The numeric part
            unit = match.group(3)  # The unit prefix
            multiplier = unit_multipliers.get(unit, 1)  # Get multiplier, default to 1
            return str(value * multiplier)  # Convert to numeric equivalent

        return re.sub(unit_pattern, unit_replacer, expression)

    updated_dict = {}

    for key, value in main_dict.items():
        if isinstance(value, str):
            # Evaluate embedded expressions in the string
            updated_value = evaluate_expression(value, replacement_dict)

            # If the evaluated value is numeric and a string, convert it to float and round
            if isinstance(updated_value, str) and re.match(r'^-?\d+(\.\d+)?$', updated_value):
                updated_dict[key] = round(float(updated_value), precision)
            else:
                updated_dict[key] = updated_value
        elif isinstance(value, (int, float)):
            # Round numeric values directly
            updated_dict[key] = round(value, precision)
        else:
            # If the value is not a string or numeric type, leave it as-is
            updated_dict[key] = value

    return updated_dict


def netlist_to_graph(netlist_content, values):
    """
    Converts a netlist to a graph representation using NetworkX.

    Args:
        netlist_content (list of str): List of strings representing the lines of a netlist.
        values (dict): A dictionary containing replacement values for attributes.

    Returns:
        networkx.MultiGraph: A MultiGraph representation of the netlist, 
                             where nodes represent circuit nodes and edges represent components.
    """
    # Initialize a MultiGraph to allow multiple edges between nodes
    G = nx.MultiGraph()

    # Preprocess the netlist to handle multi-line entries
    netlist_content = preprocess_netlist(netlist_content)

    # Iterate over each line in the netlist
    for line in netlist_content:
        # Skip comments and blank lines
        if line.startswith('//') or line.strip() == '':
            continue

        # Parse the line into component name, nodes, type, and attributes
        comp_name, nodes, comp_type, attributes = parse_netlist_line(line)

        # Replace and evaluate attribute values based on the provided dictionary
        updated_attributes = replace_values_and_evaluate(attributes, values)

        # Categorize the updated attributes into numeric, parametric, and computation dictionaries
        numeric_dict, parametric_dict, computation_dict = categorize_attributes(updated_attributes)

        # if numeric_dict != {}:
        #     print(numeric_dict)
        # if parametric_dict != {}:
        #     print(parametric_dict)
        # if computation_dict != {}:
        #     print(computation_dict)

        # Normalize node names by removing backslashes
        for idx in range(len(nodes)):
            nodes[idx] = nodes[idx].replace('\\', '')

        # If the component name is valid, process the edges in the graph
        if comp_name:
            # Iterate over all pairs of nodes connected by the component
            for i in range(len(nodes) - 1):
                for j in range(i + 1, len(nodes)):

                    # Skip edges where both nodes are in the ground list
                    if nodes[i] in GND_LIST and nodes[j] in GND_LIST:
                        continue
                    
                    if comp_type == 'nmos' or comp_type == 'pmos':
                        # Skip invalid connections for nmos and pmos components
                        if j == 3:
                            continue
                        # Append specific labels to comp_type based on the indices (i, j)
                        if i == 0 and j == 1:
                            labeled_comp_type = f"{comp_type}_DG"
                        elif i == 0 and j == 2:
                            labeled_comp_type = f"{comp_type}_DS"
                        elif i == 1 and j == 2:
                            labeled_comp_type = f"{comp_type}_GS"
                    elif comp_type == 'balun':
                        # Skip invalid connections for balun components
                        if i != 0:
                            break
                        # Append specific labels to comp_type
                        elif j == 1:
                            labeled_comp_type = f"{comp_type}_D1"
                        elif j == 2:
                            labeled_comp_type = f"{comp_type}_D2"
                    else:
                        labeled_comp_type = comp_type   # No change for other cases

                    # Normalize ground node names to 'GND'
                    if nodes[j] in GND_LIST:
                        nodes[j] = 'GND'
                    if nodes[i] in GND_LIST:
                        nodes[i] = 'GND'

                    # Add an edge to the graph with all relevant attributes
                    G.add_edge(
                        nodes[j], 
                        nodes[i], 
                        component=labeled_comp_type,  # Type of the component (e.g., nmos, resistor)
                        name=comp_name,  # Name of the component
                        numeric_attrs=numeric_dict,  # Numeric attributes of the component
                        parametric_attrs=parametric_dict,  # Parametric attributes of the component
                        computing_attrs=computation_dict  # Computed attributes of the component
                    )

    # Return the constructed graph
    return G


# Main function to process all netlists
def process_all_netlists(root_dir):
    """Iterates through all netlists in dataset and saves graphs."""      
    for circuit_type in os.listdir(root_dir):
        circuit_path = os.path.join(root_dir, circuit_type)

        if not os.path.isdir(circuit_path):
            continue  # Skip non-folder files
        
        print(f"Processing circuit type: {circuit_type}")

        for subfolder in os.listdir(circuit_path):
            subfolder_path = os.path.join(circuit_path, subfolder)

            if not os.path.isdir(subfolder_path):
                continue  # Skip files like .DS_Store

            netlist_path = os.path.join(subfolder_path, "netlist")
            csv_path = os.path.join(subfolder_path, "dataset.csv")

            if not os.path.exists(netlist_path) or not os.path.exists(csv_path):
                print(netlist_path)
                print(csv_path)
                print("  Data is not complete!")
                continue  # Skip folders without required files

            print(f"  Processing subfolder: {subfolder}")

            graph_file = os.path.join(subfolder_path, f"graph.json")
                
            # Convert netlist to PyG graph
            graph_data, node_mapping, edge_attr_dict = netlist_to_pyg_graph(subfolder_path)
            print("Nodes:", graph_data.num_nodes, "Edges:", graph_data.num_edges)

            # Save graph
            save_graph(graph_data, node_mapping, edge_attr_dict, graph_file)
            print(f"    Saved graph to {graph_file}")


def netlist_to_pyg_graph(data_path):
    values = load_values(data_path)
    netlist = load_netlist(data_path)
    netlist_graph = netlist_to_graph(netlist, values)
    pyg_data, node_mapping, edge_attr_dict = networkx_to_pyg(netlist_graph)
    return pyg_data, node_mapping, edge_attr_dict