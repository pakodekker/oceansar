import re
import numpy as np
import ast

class Section():
    def __init__(self):
        self.elements = []


class ConfigFile():
    """ Class to read/write configuration files.
        Variables are loaded as class variables, with section support. Supported features are
        described in the following table:

        ================  ============================================================================
        Feature           Description
        ----------------  ----------------------------------------------------------------------------
        Comments          * Lines starting with '#' or ';' will considered comments and thus ignored
                          * In-line comments supported (ex. ``variable=test  ;Description``)
                          * Comments will **NOT** be written back when saving.
        Sections          Defined with brackets [].
        Variable formats  * Strings
                            - path=/example/string/variable
                          * Integers
                            - int_var=2
                          * Floating point numbers
                            - float_var=2.35e6
                          * Booleans
                            - bool_var=True
                          * Arrays (1-D, Floats or strings, or Booleans)
                            - array_flt = [1.22, 3.43]
                            - array_str = ['a','b']
                            - array_bool = [True, False, True]
        ================  ============================================================================

        Example::

            # Comment
            [Section1]
            string_var=Example
            boolean_var=True
            array_str=['a','b']

            ; Comment
            [Section2]
            int_var=2
            float_var=2.0
            array_flt=[1.22,3.55]

        would be accessed as::

            class_inst_name.Section1.string_var
            class_inst_name.Section1.boolean_var
            class_inst_name.Section1.array_str
            class_inst_name.Section2.int_var
            class_inst_name.Section2.float_var
            class_inst_name.Section2.array_flt

        :param cfg_file_name: Route of parameter file
    """

    def __init__(self, cfg_file_name, str_mode=False):
        self.cfg_file_name = cfg_file_name
        self.str_mode = str_mode
        self.read()

    def read(self):
        """ Read file content

            .. note::
                Automatically called when class is initialized
        """

        self.sections = [self]
        self.elements = []
        current_section = self

        file = open(self.cfg_file_name, 'r')

        for line in file:

            # Jump blank & comment lines
            if re.search(r'(^[;#].*$)|(^[\s\t\r\n]*$)', line): continue

            # Clean line: comments & left/right strip
            line = line.rstrip().lstrip()
            line = re.sub(r'[;#].*$','', line)

            # Section line
            s = re.search(r'\[(\w+)\]', line)
            if s:
                vars(self)[s.group(1)] = Section()
                self.sections.append(s.group(1))

                current_section = vars(self)[s.group(1)]

            # Variable line
            else:

                s = re.search(r'(\w+)\s*=\s*([\s\w\s\\\:\.,/\'\-\[\]]+)', line)

                if s:
                    # String mode: no conversion to 'native' type
                    if self.str_mode:
                        vars(current_section)[s.group(1)] = s.group(2)
                    else:
                        # Integer
                        if s.group(2).isdigit():
                            vars(current_section)[s.group(1)] = int(s.group(2))
                        # Booleans
                        elif (s.group(2) == 'True') or (s.group(2) == 'False'):
                            vars(current_section)[s.group(1)] = True if s.group(2) == 'True' else False
                        # Arrays
                        elif s.group(2)[0] == '[':
                            # String array
                            if s.group(2)[1] == '\'':
                                s_n = s.group(2).replace('\'', '')
                                vars(current_section)[s.group(1)] = s_n[1:-1].split(',')
                            elif ((s.group(2)[1]) == 'T' or
                                  (s.group(2)[1] == 'F')):
                                # Boolean
                                tmp = s.group(2)[1:-1].split(',')
                                btmp = np.ones(len(tmp), dtype=bool)
                                for ind in range(len(tmp)):
                                    # Convert to boolean, after removing
                                    btmp[ind] = ast.literal_eval(tmp[ind].
                                                                 strip())
                                vars(current_section)[s.group(1)] = btmp

                            # Float array
                            else:
                                vars(current_section)[s.group(1)] = np.array(s.group(2)[1:-1].split(',')).astype(float)
                        # Floats & strings
                        else:
                            try:
                                vars(current_section)[s.group(1)] = float(s.group(2))
                            except ValueError:
                                vars(current_section)[s.group(1)] = s.group(2)

                    # Keep variable ordering
                    current_section.elements.append(s.group(1))

        file.close()

    def save(self, alternate_file=None, blank_lines=True):
        """ Saves configuration file

            :param alternate_file: Alternate path to save file
            :param blank_lines: Add a blank line after each section
        """

        file = open(alternate_file if alternate_file else self.cfg_file_name, 'w')

        for section in self.sections:

            # Check for out-of-section parameters
            if section is self:
                sect_inst = self
            else:
                file.write('[%s]\n' % section)
                sect_inst = vars(self)[section]

            for element in sect_inst.elements:

                element_value = vars(sect_inst)[element]

                if type(element_value) is int:
                    file.write('%s=%d\n' % (element, element_value))

                elif type(element_value) is float:
                    file.write('%s=%.2f\n' % (element, element_value))

                elif type(element_value) is bool:
                    file.write('%s=%s\n' % (element, 'True' if element_value else 'False'))

                elif (type(element_value) is str) or (type(element_value) is unicode):
                    file.write('%s=%s\n' % (element, element_value))

                elif type(element_value) is list:
                    file.write('%s=['%element + ','.join('\'%s\''%f for f in element_value) + ']\n')

                elif type(element_value) is np.ndarray:
                    file.write('%s=['%element + ','.join('%.2f'%f for f in element_value) + ']\n')

                else:
                    print(type(element_value))
                    print(element_value)
                    raise TypeError('Variable type not supported')

            if blank_lines:
                file.write('\n')


        file.close()
