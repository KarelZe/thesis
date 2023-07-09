# Check style:
# https://matt.might.net/articles/shell-scripts-for-passive-voice-weasel-words-duplicates/
proof:
	echo "weasel words: "
	sh ../../src/otc/utils/weasel.sh *.tex
	echo
	echo "passive voice: "
	sh ../../src/otc/utils/passive.sh *.tex
	echo
	echo "duplicates: "
	perl ../../src/otc/utils/dups.pl *.tex