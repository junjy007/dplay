"""
Deploy experiment.
"""
import json
import re
import numpy as np


class CellSectionParser(object):
    def __init__(self, start_token, end_token=None):
        self.active = False
        self.start_token = start_token
        self.end_token = start_token + '-END' if end_token is None \
            else end_token

    def parse(self, cell):
        if self.active:
            if cell['cell_type'] == 'markdown':
                if len(cell['source']) > 0 and cell['source'][0].startswith(self.end_token):
                    self.active = False
                return []
            elif cell['cell_type'] == 'code':
                cell_src = [
                               l_ for l_ in cell['source'] if not l_.startswith('%')
                               ] + ['\n', ] * 2
                return cell_src
        else:
            if cell['cell_type'] == 'markdown':
                if len(cell['source']) > 0 and cell['source'][0].startswith(self.start_token):
                    self.active = True
            return []

    def deactive(self):
        """
        Can be deactivated implicitly by starting another section
        """
        self.active = False


class ComponentParser(CellSectionParser):
    """
    Deal with 'Component' section, collect codes for component classes.
    """

    def __init__(self, components):
        """
        :param components: component-name -> component class, e.g.
          components['Encoder'] = DeepConvEncoder.
        :type components: dict
        """
        super(ComponentParser, self).__init__('# Components')
        self.clsnames = [components[k_].__name__ for k_ in components]

    def parse(self, cell):
        src = super(ComponentParser, self).parse(cell)
        if len(src) > 0:
            for cname in self.clsnames:
                has, imp_fname = self.has_def(cname, src)
                if has:
                    return src, imp_fname
        return [], None

    def has_def(self, cls_name, src):
        """
        :param clsname: class name to find definition
        :param src: list of lines of code
        :return: True if the src contains definition
        """
        for s_ in src:
            pl = re.split('\W+', s_)
            match1 = pl[0] == 'class' and pl[1] == cls_name
            match2 = pl[0] == 'from' and pl[2] == 'import' and pl[3] == cls_name
            if match1:
                # print "Find def", s_
                return True, None
            if match2:
                # print "Find import", s_
                return True, pl[1]
        return False, None



def collect_source_code(nb_fname, framework_name, components):
    """
    From a Python notebook, take working cells, generate experiment Python script.
    :param framework_name:
    :param nb_fname:
    :param components:
    :return: source, dependencies, source: plain text of source code,
      dependencies: full file name list
    """

    with open(nb_fname, 'r') as f:
        nb = json.load(f)

    src = []

    pre_parser = CellSectionParser('# Prerequisites')
    f_parser = CellSectionParser('# Framework-' + framework_name)
    c_parser = ComponentParser(components)
    for ce in nb['cells']:
        par0 = pre_parser.parse(ce)
        par1 = f_parser.parse(ce)
        par2, fn = c_parser.parse(ce)
        # src.extend()
        # src.extend(f_parser.parse(ce))
        acts = np.asarray([pre_parser.active, f_parser.active, c_parser.active], dtype=int)
        assert acts.sum() <= 1, "At most be in one stage"
        src.extend(par0)
        src.extend(par1)
        src.extend(par2)
        #print acts, len(par0), len(par1), len(par2)

    return src  #TODO return dependencies

if __name__ == '__main__':
    class RLNet:
        pass
    class DeepConvEncoder:
        pass
    collect_source_code('nb01_reinforce_framework.ipynb', 'F1',
                        {'1': RLNet, '2': DeepConvEncoder})