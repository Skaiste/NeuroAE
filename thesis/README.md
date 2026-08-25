# Thesis LaTeX project

Compile from this directory with:

```sh
latexmk -pdf main.tex
```

The main document is `main.tex`; chapter files live in `chapters/`, front matter in `frontmatter/`, and bibliography/appendix entry points in `backmatter/`.

SVG figures are converted to PDF by Inkscape during compilation. Install
Inkscape and build with shell escape enabled:

```sh
latexmk -pdf -shell-escape main.tex
```

Some draw.io SVG exports use XHTML `foreignObject` labels, which Inkscape cannot
export reliably. Convert those labels to native SVG text before compiling:

```sh
cd thesis
python3 scripts/drawio_foreignobject_to_text.py \
  ../figures/linearAEarchitecture.svg \
  ../figures/linearAEarchitecture_latex.svg
```
