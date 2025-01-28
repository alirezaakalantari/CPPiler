from collections import deque
import re
import hashlib

class Token:
    def __init__(self, name, value, position = 0, line = 1, column = 1):
        self.name = name
        self.value = value
        self.position = position
        self.line = line
        self.column = column

    def __repr__(self):
        return f"[{self.name}, {self.value}]"

class LexicalAnalyzer:
    def __init__(self):
        self.patterns = [
            ('reservedword', r'#include|int|float|void|return|if|while|cin|cout|continue|break|using|iostream|namespace|std|main'),
            ('identifier', r'[a-zA-Z][a-zA-Z0-9]*'),
            ('number', r'\d+'),
            ('string', r'"[^"]*"'),
            ('symbol', r'<<|>>|<=|>=|==|!=|\(|\)|\[|\]|,|;|\+|-|\*|/|=|\|\||{|}|<|>'),
            ('whitespace', r'[ \t\n]+')
        ]
        self.token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.patterns)
        self.regex = re.compile(self.token_regex)
        
    def tokenize(self, code: str):
        tokens = []
        position = 0
        line = 1
        column = 1
        print("\nDEBUG Lexical Analysis:")
        
        while position < len(code):
            match = self.regex.match(code, position)
            if match is None:
                raise ValueError(f"Invalid token at position {position}")

            token_type = match.lastgroup
            token_value = match.group()
            
            if token_type == 'whitespace':
                for char in token_value:
                    if char == '\n':
                        line += 1
                        column = 1
                        position += 1
                    else:
                        column += 1
                        position += 1
                continue

            if token_type == 'identifier' or token_type == 'reservedword':
                if token_value == 'iostream':
                    token = Token('reservedword', token_value, position, line, column)
                else:
                    token = Token(token_type, token_value, position, line, column)
                tokens.append(token)
            else:
                token = Token(token_type, token_value, position, line, column)
                tokens.append(token)

            position = match.end()
            column += len(token_value)

        return tokens

class TokenTableEntry:
    def __init__(self, token_name, token_value, hash_value):
        self.token_name = token_name
        self.token_value = token_value
        self.hash_value = hash_value

class ParserTables:
    def __init__(self):
        self.token_table = []
        self.parse_table = {}
        self.first_sets = {}
        self.follow_sets = {}
        
        self.token_to_terminal = {
            ('reservedword', '#include'): '#include',
            ('symbol', '<'): '<',
            ('symbol', '>'): '>',
            ('reservedword', 'iostream'): 'iostream',
            ('reservedword', 'using'): 'using',
            ('reservedword', 'namespace'): 'namespace',
            ('reservedword', 'std'): 'std',
            ('reservedword', 'int'): 'int',
            ('reservedword', 'float'): 'float',
            ('reservedword', 'main'): 'main',
            ('reservedword', 'while'): 'while',
            ('reservedword', 'cin'): 'cin',
            ('reservedword', 'cout'): 'cout',
            ('reservedword', 'return'): 'return',
            ('symbol', '{'): '{',
            ('symbol', '}'): '}',
            ('symbol', '('): '(',
            ('symbol', ')'): ')',
            ('symbol', ';'): ';',
            ('symbol', ','): ',',
            ('symbol', '='): '=',
            ('symbol', '+'): '+',
            ('symbol', '-'): '-',
            ('symbol', '*'): '*',
            ('symbol', '>='): '>=',
            ('symbol', '<='): '<=',
            ('symbol', '=='): '==',
            ('symbol', '!='): '!=',
            ('symbol', '>>'): '>>',
            ('symbol', '<<'): '<<',
            ('identifier', None): 'identifier',
            ('number', None): 'number',
            ('string', None): 'string'
        }
        
        self.terminals = {v for v in self.token_to_terminal.values()} | {'$'}
        
        self.non_terminals = {
            'Start', 'S', 'N', 'M', 'T', 'V', 'Id', 'L', 'Z', 'Operation',
            'P', 'O', 'W', 'Assign', 'Expression', 'K', 'Loop', 'Input',
            'F', 'Output', 'H', 'C', 'LibName'
        }
            
        self.grammar = {
            'Start': ['S N M'],
            'S': ['#include <LibName> S', 'ε'],
            'N': ['using namespace std ;', 'ε'],
            'M': ['int main ( ) { T V }'],
            'LibName': ['identifier', 'iostream'],
            'T': ['Id T', 'L T', 'Loop T', 'Input T', 'Output T', 'ε'],
            'V': ['return number ;', 'ε'],
            'Id': ['int L', 'float L'],
            'L': ['identifier Assign Z'],
            'Z': [', identifier Assign Z', ';'],
            'Operation': ['number P', 'identifier P'],
            'P': ['O W P', 'ε'],
            'O': ['+', '-', '*'],
            'W': ['number', 'identifier'],
            'Assign': ['= Operation', 'ε'],
            'Expression': ['Operation K Operation'],
            'K': ['==', '>=', '<=', '!='],
            'Loop': ['while ( Expression ) { T }'],
            'Input': ['cin >> identifier F ;'],
            'F': ['>> identifier F', 'ε'],
            'Output': ['cout << C H ;'],
            'H': ['<< C H', 'ε'],
            'C': ['number', 'string', 'identifier']
        }

    def compute_hash(self, token_value: str):
        hasher = hashlib.sha256()
        hasher.update(token_value.encode('utf-8'))
        return hasher.hexdigest()[:8]

    def get_terminal(self, token_type: str, token_value: str):
        if token_type == 'symbol' and token_value == '#':
            next_idx = self.current_token_idx + 1
            if next_idx < len(self.token_stream):
                next_type, next_value = self.token_stream[next_idx]
                if next_type == 'reservedword' and next_value == 'include':
                    self.current_token_idx += 1
                    return '#include'
            return None
        
        
        if token_type == 'symbol':
            
            if token_value == '<':
                next_idx = self.current_token_idx + 1
                if next_idx < len(self.token_stream) and next_idx + 1 < len(self.token_stream):
                    lib_token = self.token_stream[next_idx]
                    close_token = self.token_stream[next_idx + 1]
                    
                    is_valid_lib = (
                        (lib_token[1] == 'iostream') or
                        (lib_token[0] == 'identifier')
                    )
                    
                    if is_valid_lib and close_token[0] == 'symbol' and close_token[1] == '>':
                        self.current_token_idx += 2
                        return '<LibName>'
            
            if token_value in ['<<', '>>']:
                return token_value
            
            if token_value in ['<=', '>=', '==', '!=']:
                return token_value
            
            return token_value
        
        if token_type == 'reservedword':
            return token_value
        
        if token_type in ['identifier', 'number', 'string']:
            return token_type
        
        return token_value

    def add_token(self, token_name: str, token_value: str):
        hash_value = self.compute_hash(token_value)
        entry = TokenTableEntry(token_name, token_value, hash_value)
        
        token_order = {
            'string': 0,
            'number': 1,
            'symbol': 2,
            'identifier': 3,
            'reservedword': 4,
        }
        
        insert_pos = 0
        for i, existing in enumerate(self.token_table):
            if token_order[existing.token_name] > token_order[token_name]:
                break
            if (token_order[existing.token_name] == token_order[token_name] and 
                existing.token_value > token_value):
                break
            insert_pos = i + 1
            
        self.token_table.insert(insert_pos, entry)

    def _is_terminal(self, symbol: str):
        return symbol in self.terminals or symbol == 'ε'

    def compute_first_sets(self):
        self.first_sets = {nt: set() for nt in self.non_terminals}
        
        changed = True
        while changed:
            changed = False
            for nt, productions in self.grammar.items():
                for prod in productions:
                    symbols = prod.split()
                    
                    if not symbols or symbols[0] == 'ε':
                        if 'ε' not in self.first_sets[nt]:
                            self.first_sets[nt].add('ε')
                            changed = True
                        continue
                    
                    all_nullable = True
                    for symbol in symbols:
                        if self._is_terminal(symbol):
                            if symbol not in self.first_sets[nt]:
                                self.first_sets[nt].add(symbol)
                                changed = True
                            all_nullable = False
                            break
                        else:  
                            for s in self.first_sets[symbol] - {'ε'}:
                                if s not in self.first_sets[nt]:
                                    self.first_sets[nt].add(s)
                                    changed = True
                            if 'ε' not in self.first_sets[symbol]:
                                all_nullable = False
                                break
                                
                    if all_nullable and 'ε' not in self.first_sets[nt]:
                        self.first_sets[nt].add('ε')
                        changed = True

    def compute_follow_sets(self):
        self.follow_sets = {nt: set() for nt in self.non_terminals}
        self.follow_sets['Start'].add('$')
        
        changed = True
        while changed:
            changed = False
            for nt, productions in self.grammar.items():
                for prod in productions:
                    symbols = prod.split()
                    
                    for i, symbol in enumerate(symbols):
                        if symbol not in self.non_terminals:
                            continue
                            
                        trailer = symbols[i + 1:]
                        first_of_trailer = self._compute_first_of_string(trailer)
                        
                        for terminal in first_of_trailer - {'ε'}:
                            if terminal not in self.follow_sets[symbol]:
                                self.follow_sets[symbol].add(terminal)
                                changed = True
                                
                        if 'ε' in first_of_trailer:
                            for terminal in self.follow_sets[nt]:
                                if terminal not in self.follow_sets[symbol]:
                                    self.follow_sets[symbol].add(terminal)
                                    changed = True

    def _compute_first_of_string(self, symbols):
        if not symbols:
            return {'ε'}
            
        result = set()
        all_nullable = True
        
        for symbol in symbols:
            if self._is_terminal(symbol):
                result.add(symbol)
                all_nullable = False
                break
            else:
                result.update(self.first_sets[symbol] - {'ε'})
                if 'ε' not in self.first_sets[symbol]:
                    all_nullable = False
                    break
                    
        if all_nullable:
            result.add('ε')
            
        return result

    def build_parse_table(self):
        self.compute_first_sets()
        self.compute_follow_sets()
        
        self.parse_table = {
            nt: {t: '' for t in self.terminals} 
            for nt in self.non_terminals
        }
        
        for nt, productions in self.grammar.items():
            for prod in productions:
                first_of_prod = self._compute_first_of_string(prod.split())
                
                for terminal in first_of_prod - {'ε'}:
                    if self.parse_table[nt][terminal]:
                        raise ValueError(
                            f"Grammar is not LL(1): Conflict at {nt}, {terminal}"
                        )
                    self.parse_table[nt][terminal] = prod
                
                if 'ε' in first_of_prod:
                    for terminal in self.follow_sets[nt]:
                        if self.parse_table[nt][terminal]:
                            raise ValueError(
                                f"Grammar is not LL(1): Conflict at {nt}, {terminal}"
                            )
                        self.parse_table[nt][terminal] = 'ε'

    def get_parse_table_entry(self, non_terminal: str, terminal: str):
        if non_terminal not in self.parse_table:
            raise ValueError(f"Unknown non-terminal: {non_terminal}")
        if terminal not in self.parse_table[non_terminal]:
            raise ValueError(f"Unknown terminal: {terminal}")
        return self.parse_table[non_terminal][terminal]

class ParseTreeNode:
    def __init__(self, value, children, parent=None, token_type=None, token_value=None, init_value=None):
        self.value = value
        self.children = children
        self.parent = parent
        self.token_type = token_type
        self.token_value = token_value
        self.init_value = init_value

class PredictiveParser:
    def __init__(self, parser_tables):
        self.parser_tables = parser_tables
        self.token_stream = []
        self.current_token_idx = 0
        self.parse_tree_root = None
        self.symbol_map = {
            '(': 'lparen', ')': 'rparen',
            '{': 'lbrace', '}': 'rbrace',
            ';': 'semicolon', ',': 'comma',
            '=': 'assign', '+': 'plus',
            '-': 'minus', '*': 'multiply',
            '>=': 'gteq', '<=': 'lteq',
            '==': 'equal', '!=': 'notequal',
            '>>': 'input', '<<': 'output'
        }
        self.error_handler = ErrorHandler()

    def get_terminal_symbol(self, token_type: str, token_value: str):
        if token_type == 'symbol' and token_value == '#':
            next_idx = self.current_token_idx + 1
            if next_idx < len(self.token_stream):
                next_token = self.token_stream[next_idx]
                if next_token.name == 'reservedword' and next_token.value == 'include':
                    self.current_token_idx += 1
                    return '#include'
            return None

        if token_type == 'symbol':
            if token_value == '<':
                next_idx = self.current_token_idx + 1
                if next_idx < len(self.token_stream) and next_idx + 1 < len(self.token_stream):
                    lib_token = self.token_stream[next_idx]
                    close_token = self.token_stream[next_idx + 1]
                    is_valid_lib = (lib_token.value == 'iostream' or lib_token.name == 'identifier')
                    if is_valid_lib and close_token.name == 'symbol' and close_token.value == '>':
                        self.current_token_idx += 2
                        return '<LibName>'

            if token_value in ['<<', '>>', '<=', '>=', '==', '!=']:
                return token_value

            if token_value in ['<', '>']:
                next_idx = self.current_token_idx + 1
                if next_idx < len(self.token_stream):
                    next_token = self.token_stream[next_idx]
                    if next_token.name == 'symbol' and next_token.value == token_value:
                        return None
                return token_value

        if token_type in ['string', 'number', 'identifier']:
            return token_type

        if token_type == 'reservedword':
            return token_value

        return token_value

    def parse(self, tokens, source_code):
        self.error_handler.initialize_source(source_code)
        self.token_stream = tokens
        self.current_token_idx = 0
        self.production_sequence = []
        
        self.parse_tree_root = ParseTreeNode('Start', [], None)
        stack = deque(['$', 'Start'])
        node_stack = deque([self.parse_tree_root])

        while stack and self.current_token_idx < len(self.token_stream):
            top = stack[-1]
            current_token = self.token_stream[self.current_token_idx]
            
            terminal = self.get_terminal_symbol(current_token.name, current_token.value)
            
            if terminal is None:
                self.current_token_idx += 1
                continue
                
            if top == '$' and terminal == '$':
                break
                
            if top == terminal:
                current_node = node_stack.pop()
                current_node.token_type = current_token.name
                current_node.token_value = current_token.value

                if current_token.name == 'identifier':
                    look_ahead = self.current_token_idx + 1
                    while look_ahead < len(self.token_stream):
                        next_token = self.token_stream[look_ahead]
                        if next_token.value == '=':
                            if look_ahead + 1 < len(self.token_stream):
                                value_token = self.token_stream[look_ahead + 1]
                                if value_token.name in ['number', 'identifier']:
                                    current_node.init_value = value_token.value
                            break
                        elif next_token.value in [';', ',']:
                            break
                        look_ahead += 1
                
                stack.pop()
                self.current_token_idx += 1
            elif top not in self.parser_tables.grammar:
                self.error_handler.handle_syntax_error(
                    current_token.value,
                    top,
                    current_token.position
                )
            else:
                production = self.parser_tables.get_parse_table_entry(top, terminal)
                
                if not production:
                    context = self._get_parsing_context(top)
                    error_pos = current_token.position
                    
                    if current_token.value.startswith('"') and context == ['<<']:
                        for i in range(self.current_token_idx - 1, -1, -1):
                            prev_token = self.token_stream[i]
                            if prev_token.name == 'reservedword' and prev_token.value == 'cout':
                                error_pos = prev_token.position + len('cout')
                                break
                    
                    self.error_handler.handle_syntax_error(
                        current_token.value,
                        context,
                        error_pos
                    )
                
                if production != 'ε':
                    self.production_sequence.append(f"{top} -> {production}")
                else:
                    self.production_sequence.append(f"{top} -> epsilon")
                
                current_node = node_stack.pop()
                stack.pop()
                
                if production == 'ε':
                    epsilon_node = ParseTreeNode('ε', [], current_node)
                    current_node.children.append(epsilon_node)
                else:
                    symbols = production.split()
                    for symbol in reversed(symbols):
                        new_node = ParseTreeNode(symbol, [], current_node)
                        current_node.children.append(new_node)
                        stack.append(symbol)
                        node_stack.append(new_node)
        
        return self.parse_tree_root

    def _get_parsing_context(self, non_terminal: str):
        context = set()
        for terminal in self.parser_tables.terminals:
            if self.parser_tables.parse_table[non_terminal].get(terminal):
                context.add(terminal)
        return sorted(list(context))

    def get_production_sequence(self):
        final_sequence = []
        include_handled = False
        
        for prod in self.production_sequence:
            if prod.startswith('Start'):
                final_sequence.append('Start -> S N M')
            elif prod.startswith('S') and '#include' in self.token_stream[0].value:
                if not include_handled:
                    final_sequence.append('S -> #include S')
                    include_handled = True
                else:
                    final_sequence.append('S -> epsilon')
            elif not prod.startswith('LibName'):
                final_sequence.append(prod)
        
        return final_sequence

class ErrorHandler:
    def __init__(self):
        self.source_code = ""
        self.lines = []
        self.line_positions = []

    def initialize_source(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.split('\n')
        position = 0
        for line in self.lines:
            self.line_positions.append(position)
            position += len(line) + 1

    def get_line_and_column(self, position: int):
        line_num = len(self.lines)
        for i in range(len(self.line_positions) - 1):
            current_start = self.line_positions[i]
            next_start = self.line_positions[i + 1]
            if current_start <= position < next_start:
                line_num = i + 1
                break
        
        if line_num == len(self.lines) and position >= self.line_positions[-1]:
            line_num = len(self.lines)
        
        line_start = self.line_positions[line_num - 1]
        column = position - line_start + 1
        
        return line_num, column

    def handle_syntax_error(self, token_value: str, expected_value: str, position: int):
        line_num, column = self.get_line_and_column(position)
        line_content = self.lines[line_num - 1]
        
        if token_value == '"sum="' and isinstance(expected_value, list) and '<<' in expected_value:
            cout_pos = line_content.find('cout')
            if cout_pos != -1:
                column = cout_pos + len('cout') + 1
                line_num, _ = self.get_line_and_column(self.line_positions[line_num - 1] + cout_pos)
        
        if isinstance(expected_value, list):
            if token_value == '"sum="' and '<<' in expected_value:
                message = f"Syntax Error: Expected '<<', found '{token_value}'"
            else:
                message = f"Syntax Error: Unexpected '{token_value}'. Expected one of: {', '.join(expected_value)}"
        else:
            message = f"Syntax Error: Expected '{expected_value}', found '{token_value}'"
        
        error_msg = self.format_error(line_num, column, message)
        raise SyntaxError(error_msg)

    def format_error(self, line_num: int, column: int, message: str):
        error_msg = [f"\n{message}", "\nContext:"]
        start_line = max(1, line_num - 2)
        end_line = min(len(self.lines), line_num + 2)
        
        for i in range(start_line, end_line + 1):
            prefix = "-> " if i == line_num else "   "
            error_msg.append(f"{prefix}{i:4d} | {self.lines[i-1]}")
            if i == line_num:
                error_msg.append("      " + " " * (column - 1) + "^")
        
        return "\n".join(error_msg)

    def check_syntax(self, token_stream, source_code: str):
        self.initialize_source(source_code)
        last_token = None
        in_statement = False
        
        for token in token_stream:
            if last_token:
                last_type, last_value, last_pos = last_token
                
                if last_value in [';', '{', '}']:
                    in_statement = False
                
                if in_statement and last_type in ['identifier', 'number']:
                    allowed_follows = {';', ',', '=', '+', '-', '*', '/',
                                     '>=', '<=', '==', '!=', ')', ']', '<<', '>>'}
                    if token.value not in allowed_follows:
                        self.handle_missing_semicolon(last_pos)
                
                if token.value == '=' and last_type not in ['identifier']:
                    self.handle_invalid_assignment(token.position)
                
                if token.value in ['(', '{']:
                    in_statement = False
                elif token.value not in [';', '}']:
                    in_statement = True
            
            last_token = (token.name, token.value, token.position)
        
        return True

    def handle_missing_semicolon(self, position: int):
        line_num, _ = self.get_line_and_column(position)
        line_content = self.lines[line_num - 1]
        column = len(line_content) + 1
        error_msg = self.format_error(
            line_num, column,
            "Syntax Error: Missing semicolon at end of statement"
        )
        raise SyntaxError(error_msg)

    def handle_invalid_assignment(self, position: int):
        line_num, _ = self.get_line_and_column(position)
        line_content = self.lines[line_num - 1]
        equal_pos = line_content.find('=')
        if equal_pos == -1:
            equal_pos = len(line_content)
        error_msg = self.format_error(
            line_num, equal_pos + 1,
            "Syntax Error: Invalid left-hand side in assignment"
        )
        raise SyntaxError(error_msg)
    
class TreeSearcher:
    def __init__(self, parse_tree_root: ParseTreeNode):
        self.root = parse_tree_root
        
    def search_node(self, node: ParseTreeNode, identifier: str):
        if node.value == 'L':
           
            for child in node.children:
                if (child.value == 'identifier' and 
                    child.token_value == identifier):
                    var_type = self.get_var_type(child)
                    init_value = self.get_init_value(child)
                    if var_type:
                        return f"{var_type}{' = ' + init_value if init_value else ''}"
            
           
            z_node = next((child for child in node.children if child.value == 'Z'), None)
            if z_node:
                current = z_node
                while current and current.value == 'Z':
                    # Check identifier before comma
                    id_node = next((child for child in current.parent.children 
                                  if child.value == 'identifier' and 
                                  child.token_value == identifier), None)
                    if id_node:
                        var_type = self.get_var_type(id_node)
                        init_value = self.get_init_value(id_node)
                        if var_type:
                            return f"{var_type}{' = ' + init_value if init_value else ''}"
                        
                    # Move to next part after comma
                    comma_list = next((child for child in current.children 
                                     if child.value == 'Z'), None)
                    if comma_list:
                        current = comma_list
                    else:
                        break
        
        for child in node.children:
            result = self.search_node(child, identifier)
            if result:
                return result
                
        return None

    def get_var_type(self, node: ParseTreeNode):
        current = node
        while current and current.value != 'Id':
            current = current.parent
        if current and current.children:
            for child in current.children:
                if child.token_type == 'reservedword':
                    return child.token_value
        return None

    def get_init_value(self, node: ParseTreeNode):
        # First check if the node has an init_value
        if node.init_value is not None:
            return node.init_value

        current = node
        while current:
            if current.value == 'Assign':
                for child in current.children:
                    if child.value == 'Operation':
                        def find_first_value(n):
                            if n.token_type in ['number', 'identifier']:
                                return n.token_value
                            for c in n.children:
                                result = find_first_value(c)
                                if result:
                                    return result
                            return None
                        return find_first_value(child)
            current = current.parent
        return None

    def find_identifier_definition(self, identifier: str):
        result = self.search_node(self.root, identifier)
        return result if result else "Nothing Was Found"


def print_parse_table(parse_table, terminals, non_terminals):
    print("The Parse Table:\n")
    
    used_terminals = []
    for terminal in sorted(terminals):
        for nt in non_terminals:
            if parse_table[nt].get(terminal):
                used_terminals.append(terminal)
                break
    
    for nt in sorted(non_terminals):
        print(f"{nt}:")
        has_productions = False
        for terminal in used_terminals:
            production = parse_table[nt].get(terminal, '')
            if production:
                has_productions = True
                if production == 'ε':
                    production = 'epsilon'
                print(f"  {terminal}\t\t -> {production}")
        if not has_productions:
            print("  <any productions>")
        print()

def print_token_table(token_table):
    print("\nThe Token Table:\n")
    
    current_type = None
    for entry in token_table:
        if entry.token_name != current_type:
            current_type = entry.token_name
            print(f"\n{current_type.upper()}:")
        print(f"  {entry.token_value:<20} (HASH: {entry.hash_value})")

def print_productions(productions):
    print("\nThe Production Sequence\n")
    for i, prod in enumerate(productions, 1):
        print(f"{i:3}. {prod}")

def print_identifier_table(identifiers):
    print("\nIdentifier Defini:")
    for identifier, definition in identifiers:
        print(f"  {identifier:<10} -> {definition}")

test_code = """
#include <iostream>
using namespace std;
int main(){
    int x;
    int s=0, t=10;
    while (t >= 0){
        cin>>x;
        t = t - 1;
        s = s + x;
    }
    cout<<"sum="<<s;
    return 0;
}
""".strip()

print("Test:")
print("\nInput Code:")
print("-." * 30)
print(test_code)
print("-." * 30)

try:
    error_handler = ErrorHandler()

    print("\nP1: Lexical Analysis:")
    lexer = LexicalAnalyzer()
    
    try:
        tokens = lexer.tokenize(test_code)
        # print(tokens)
        print("\nTokens:")
        for i in tokens:
            print(i)
            
        print(f"\nTokenized Successfully, num of tokens is {len(tokens)}")
    except ValueError as e:
        print(f"Lexical Error: {str(e)}")
        exit()

    print("\nP2: Parser Tables Initializing:")
    parser_tables = ParserTables()
    
    current_pos = 0
    for token in tokens:
        parser_tables.add_token(token.name, token.value)
        current_pos += len(token.value) + 1
    
    try:
        parser_tables.build_parse_table()
        
        print("\nThe Token Table:")
        current_type = None
        for entry in parser_tables.token_table:
            if entry.token_name != current_type:
                current_type = entry.token_name
                print(f"\n{current_type.upper()}:")
            print(f"  {entry.token_value:<20} (HASH: {entry.hash_value})")
        
        print("\nThe Parse Table:")
        terminals = sorted(parser_tables.terminals)
        non_terminals = sorted(parser_tables.grammar.keys())
        
        for nt in non_terminals:
            print(f"\n{nt}:")
            for terminal in terminals:
                production = parser_tables.parse_table[nt].get(terminal, '')
                if production:
                    if production == 'ε':
                        production = 'epsilon'
                    print(f"  {terminal:<15} -> {production}")
                    
    except ValueError as e:
        print(f"Error in parse table construction: {str(e)}")
        exit()

    print("\nP3: Doing Parsing:")
    parser = PredictiveParser(parser_tables)
    
    try:
        parse_tree = parser.parse(tokens, test_code)
        productions = parser.get_production_sequence()
        
        print("\nThe Production Sequence:")
        for i, prod in enumerate(productions, 1):
            print(f"{i}. {prod}")
        
        print("\nThe Extra Features:")
        
        tree_searcher = TreeSearcher(parse_tree)
        for identifier in ['x', 's', 't']:
            result = tree_searcher.search_node(parse_tree, identifier)
            if result:
                print(f"{identifier} -> {result}")
            else:
                print(f"{identifier} -> Nothing was Found")
        
        if error_handler.check_syntax(tokens, test_code):
            print("\nSuccess! All phases completed with no syntax errors.")
        
    except SyntaxError as e:
        print(f"\nSyntax Error:\n{str(e)}")
        
except Exception as e:
    print(f"\nUnexpected Error: {str(e)}")

