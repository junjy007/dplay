"""
Deploy experiment.
"""
import json
import re
import numpy as np
import os
import shutil


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
          NB: the values must be class-objects (NOT class instances),
          I cancelled support of string class names -- it is to enforce the
          user to provide working classes.
        :type components: dict
        """
        super(ComponentParser, self).__init__('# Components')
        self.clsnames = [components[k_].__name__ for k_ in components]

    def parse(self, cell):
        src = super(ComponentParser, self).parse(cell)
        if len(src) > 0:
            for cname in self.clsnames:
                has, imp_fname = ComponentParser.has_def(cname, src)
                if has:
                    return src, imp_fname
        return [], None

    @staticmethod
    def has_def(cls_name, src):
        """
        :param cls_name: class name to find definition
        :param src: list of lines of code
        :return: True if the src contains definition
        """
        for s_ in src:
            pl = re.split('\s+', s_)
            match1 = pl[0] == 'class' and pl[1] == cls_name
            match2 = pl[0] == 'from' and pl[2] == 'import' and pl[3] == cls_name
            if pl[0] == 'from':
                print pl
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
    dep_srcs = []  # dependency file names

    pre_parser = CellSectionParser('# Prerequisites')
    f_parser = CellSectionParser('# Framework-' + framework_name)
    c_parser = ComponentParser(components)
    for ce in nb['cells']:
        par0 = pre_parser.parse(ce)
        par1 = f_parser.parse(ce)
        par2, fn = c_parser.parse(ce)
        acts = np.asarray([pre_parser.active, f_parser.active, c_parser.active], dtype=int)
        assert acts.sum() <= 1, "At most be in one stage"
        if len(par0) > 0:
            src.extend(par0)
        if len(par1) > 0:
            src.extend(par1)
        if len(par2) > 0:
            src.extend(par2)
            if not (fn is None):
                fns = fn.split('.')
                fns[-1] += '.py'  # dealing with "from mod1.func1 import cls1"
                dep_srcs.append(fns)  # should have ["mod1", "func1.py"]

    return src, dep_srcs  # TODO return dependencies


def deploy(nb_fname, framework_name, components, running_dir):
    """
    Deploy the experiment defined by a framework.
    :param nb_fname:
    :param framework_name:
    :param components: 'component-desc': class, using which classes to
     implement the corresponding component.
    :param running_dir: packaged experiment will be put to this directory
    :return:

    """
    print "XDeploy"
    # TODO: warm start the experiment by spawning from another experiment's checkpoints
    if not os.path.exists(running_dir):
        os.mkdir(running_dir)  # Not using "makedirs":
        # User must prepare parent directory for experiment package

    runner_src, dep_src_fnames = collect_source_code(nb_fname, framework_name, components)
    print "Dependency Module Files: {}".format(dep_src_fnames)
    with open(os.path.join(running_dir, 'run.py'), 'w') as f:
        for l in runner_src:
            f.write(l)

    for fn in dep_src_fnames:
        assert isinstance(fn, list) and len(fn) > 0
        mod_fullname_s = os.path.join(*fn)
        if len(fn) > 1:  # prepare destination dir
            mod_subdir_t = os.path.join(running_dir, *fn[:-1])
            if not os.path.exists(mod_subdir_t):
                os.mkdir(mod_subdir_t)
        else:
            mod_subdir_t = running_dir
        mod_fullname_t = os.path.join(mod_subdir_t, fn[-1])
        shutil.copyfile(mod_fullname_s, mod_fullname_t)


if __name__ == '__main__':
    # noinspection PyClassHasNoInit
    class RLNet:
        pass

    # noinspection PyClassHasNoInit
    class DeepConvEncoder:
        pass


    collect_source_code('nb01_reinforce_framework.ipynb', 'F1',
                        {'1': RLNet, '2': DeepConvEncoder})
