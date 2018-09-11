# From current dir, convert all pdf to 40% size for icons

for f in $(basename -s '.pdf' *.pdf)
do
	if test $f.pdf -nt $f-icon.pdf
	then
		echo "$f.pdf -> $f-icon.pdf"
		cpdf -scale-page '0.4 0.4' $f -o $f-icon.pdf
	fi
done

