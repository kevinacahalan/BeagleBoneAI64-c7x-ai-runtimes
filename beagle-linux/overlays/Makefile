# Device-tree overlay for the C7x AI runtime DDR tensor arena.

CPP = cpp
DTC = dtc

DTSO = k3-j721e-beagleboneai64-c7x-ai.dtso
DTBO = $(DTSO:.dtso=.dtbo)

all: $(DTBO)

%.dtbo: %.dtso
	$(CPP) -nostdinc -undef -D__DTS__ -x assembler-with-cpp -o $@.dts.tmp $<
	$(DTC) -@ -O dtb -o $@ -b 0 $@.dts.tmp
	rm -f $@.dts.tmp

clean:
	rm -f $(DTBO) *.dts.tmp
