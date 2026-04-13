import os

with open('make_notebooks.py', 'r') as f:
    code = f.read()

# Extract the strings built inside make_notebooks by declaring a dummy dict 
d = {}
exec(code, d)

# Add headless logic to test plot
headless_import = "import matplotlib; matplotlib.use('Agg')\n"

gen_code = d['code_gen_params'] + "\n" + d['code_gen_setup'] + "\n" + d['code_gen_loop']
viz_code = headless_import + "\n" + d['code_viz_params'] + "\n" + d['code_viz_setup'] + "\n" + d['code_viz_loop']

print("=== SMOKE TESTING GENERATION CELL CONTENT ===")
exec(gen_code)
print("=== SMOKE TESTING VISUALIZATION CELL CONTENT ===")
exec(viz_code)
print("ALL PIPELINE NOTEBOOK CELLS EVALUATED NATIVELY WITHOUT ERROR!")
