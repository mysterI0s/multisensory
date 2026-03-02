import os

def fix_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

fix_file('src/shift_net.py', [
    (' tf.disable_v2_behavior()\\nimport numpy', '# tf disabled\\nimport numpy'),
])

fix_file('src/tfutil.py', [
    ('def add_loss_acc(self, (loss, acc), base_name, summary = False):\\n    acc = tf.stop_gradient(acc)',
     'def add_loss_acc(self, _tuple_arg, base_name, summary = False):\\n    loss, acc = _tuple_arg\\n    acc = tf.stop_gradient(acc)')
])

fix_file('src/aolib/img.py', [
    ('def sub_img_pad(im, bbox, oob = 0):\\n    x, y, w, h = bbox',
     'def sub_img_pad(im, bbox, oob = 0):\\n  x, y, w, h = bbox')
])

fix_file('src/aolib/imtable.py', [
    ('def save_helper(args):\\n    fname, x = args',
     'def save_helper(args):\\n  fname, x = args')
])

fix_file('src/aolib/sound.py', [
    ('def resample_snd(args):\\n    snd, sr = args',
     'def resample_snd(args):\\n  snd, sr = args')
])

fix_file('src/aolib/util.py', [
    ('def unstash_seed(args):\\n    py_state, np_state = args',
     'def unstash_seed(args):\\n  py_state, np_state = args')
])
