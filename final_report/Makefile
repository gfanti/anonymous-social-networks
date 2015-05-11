TEX = xelatex
BIB = bibtex

.PHONY: clean

main.pdf: main.tex references.bib
	$(TEX) main
	$(BIB) main
	$(TEX) main
	$(TEX) main

main.bib: 

clean:
	rm -rf *.log *.aux *.pdf *.bbl *.blg *.out *.synctex.gz
