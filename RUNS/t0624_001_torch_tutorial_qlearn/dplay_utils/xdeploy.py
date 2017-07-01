"""
Deploy experiment from notebooks or scripts (scripts will be copied to the running directory,
modules in sub-folders duplicating current code structure.)
Usage:
  xdeploy.py deploy <exp_src> [to <run_dir>]
"""
import json
import re
import numpy as np
import os
import shutil
import subprocess
import docopt

def get_hostname():
    p = subprocess.Popen('hostname', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    hostname = p.stdout.readlines()[0][:-1]
    return hostname

def get_project_path():
    hostname = get_hostname()
    proj_path_dist = {
        'maibu': 'local/projects/dplay',
        'DBox': 'projects/dplay',
    }
    rel_path = proj_path_dist.get(hostname, 'projects/dplay')
    full_path = os.path.join(os.environ['HOME'], rel_path)
    return full_path

def get_filename_we(fn):
    """
    Returns the filename, without path, without extension
    :return:
    """
    return os.path.splitext(os.path.split(fn)[1])[0]


def get_path(fn):
    """
    Given a filename, (relatively locatable from (pwd)), get the absolute
    path to the file.
    :return:
    """
    return os.path.abspath(os.path.split(fn)[0])


def load_test_config(cfname, debug_info=False):
    """
    Load experiment config from specified JSON file.
    :param cfname: JSON filename, ext-name WILL BE IGNORED. So
      if using the same filename as the experiment script, this function
      can be called simply using load_test_config(__file__), where
      experiment script python filename will be used to infer the
      configuration filename.
    :return:
    """

    fn_ = get_filename_we(cfname)
    pth_ = get_path(cfname)
    if debug_info:
        print "Starting Experiment at {}\n\t{}".format(pth_, fn_)
    conf_file = os.path.join(pth_, fn_ + '.json')
    try:
        with open(conf_file, 'r') as f:
            conf = json.load(f)
        if debug_info:
            print "with following settings ..."
            print conf
    except:
        print "Failed to load experiment settings from {}".format(conf_file)
        exit(-1)
    return conf



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
            match2 = pl[0] == 'from' and pl[2] == 'import' # and pl[3] == cls_name
            # take all "import" statements from component section
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


def copypytree(sdir, ddir):
    for item in os.listdir(sdir):
        s = os.path.join(sdir, item)
        d = os.path.join(ddir, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                os.mkdir(d)
            copypytree(s, d)
        elif os.path.splitext(item)[1]=='.py':
            shutil.copy2(s, d)

def deploy_nb(nb_fname, framework_name, components, running_dir):
    """
    Deploy the experiment defined by a framework in an experimental notebook.
    :param nb_fname:
    :param framework_name:
    :param components: 'component-desc': class, using which classes to
     implement the corresponding component.
    :param running_dir: packaged experiment will be put to this directory
    :return:

    """
    # TODO: warm start the experiment by spawning from another experiment's checkpoints
    if not os.path.exists(running_dir):
        os.mkdir(running_dir)  # Not using "makedirs":
        # User must prepare parent directory for experiment package

    runner_src, dep_src_fnames = collect_source_code(nb_fname, framework_name, components)
    # print "Dependency Module Files: {}".format(dep_src_fnames)
    with open(os.path.join(running_dir, 'run.py'), 'w') as f:
        for l in runner_src:
            f.write(l)

    nb_dir = os.path.split(nb_fname)[0]
    if len(nb_dir) == 0:
        nb_dir = '.'
    copypytree(nb_dir, running_dir)
    # for fn in dep_src_fnames:
    #     assert isinstance(fn, list) and len(fn) > 0
    #     mod_fullname_s = os.path.join(*fn)
    #     if len(fn) > 1:  # prepare destination dir
    #         mod_subdir_t = os.path.join(running_dir, *fn[:-1])
    #         if not os.path.exists(mod_subdir_t):
    #             os.makedirs(mod_subdir_t)
    #     else:
    #         mod_subdir_t = running_dir
    #     mod_fullname_t = os.path.join(mod_subdir_t, fn[-1])
    #     shutil.copyfile(mod_fullname_s, mod_fullname_t)
    #     init_file_s = os.path.join(os.path.join(*fn[:-1]), '__init__.py')
    #     init_file_t = os.path.join(running_dir, init_file_s)
    #     if not os.path.exists(init_file_t):
    #         shutil.copyfile(init_file_s, init_file_t)

def deploy_sc(experiment_src, running_dir=''):
    """
    Deploy an experiment script. (generally summerised from notebook experiment)
    :return:
    """
    conf = load_test_config(experiment_src)
    test_str = get_filename_we(experiment_src)
    if len(running_dir) == 0:
        bdir = get_project_path()
        if len(test_str) > 9:
            test_str_d = test_str[:9]
        else:
            test_str_d = test_str
        running_dir = os.path.join(bdir, 'RUNS', test_str_d)
        print "Will deploy at {}".format(running_dir)

    if not os.path.exists(running_dir):
        os.mkdir(running_dir)

    copypytree('.', running_dir)
    shutil.copy2(experiment_src, running_dir)

    # writing json, instead of copying, we have chance to manipulate
    # options when deploying.
    dest_conf_file = os.path.join(running_dir, test_str+'.json')
    with open(dest_conf_file, 'w') as f:
        json.dump(conf, f, indent=2)


def test_deploy_nb():
    # noinspection PyClassHasNoInit
    class RLNet:
        pass
    # noinspection PyClassHasNoInit
    class DeepConvEncoder:
        pass
    collect_source_code('nb01_reinforce_framework.ipynb', 'F1',
                        {'1': RLNet, '2': DeepConvEncoder})

if __name__ == '__main__':
    opts = docopt.docopt(__doc__)
    running_dir = opts['<run_dir>'] if opts['to'] else ''
    deploy_sc(opts['<exp_src>'], running_dir)

