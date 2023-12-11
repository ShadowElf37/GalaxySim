import os.path

def render(string, namespace):
    import re
    for kw in set(re.findall(r'{{(.[^\}]*)}}', string)):
        string = string.replace('{{' + kw + '}}', str(eval(kw, namespace)))
    return string

def load_cl(fname, namespace):
    rendered_code = render(open(fname).read(), namespace)
    name, ext = os.path.splitext(fname)
    with open(name+'_rendered'+ext, 'w') as f:
        f.write(rendered_code)
    return rendered_code