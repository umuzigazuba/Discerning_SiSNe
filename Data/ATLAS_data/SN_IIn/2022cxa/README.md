# SN 2022cxa Light Curve Cleaning and Averaging

The ATLAS SN light curves are separated by filter (orange and cyan) and labelled as such in the file name. Averaged light curves contain an additional number in the file name that represents the MJD bin size used. Control light curves are located in the "controls" subdirectory and follow the same naming scheme, only with their control index added after the SN name.

The following details the file names for each of the light curve versions:
	- SN light curves: 2022cxa.o.lc.txt and 2022cxa.c.lc.txt
	- Averaged light curves: 2022cxa.o.1.00days.lc.txt and 2022cxa.c.1.00days.lc.txt
	- Control light curves, where X=001,...,002: 2022cxa_iX.o.lc.txt and 2022cxa_iX.c.lc.txt

The following summarizes the hex values in the "Mask" column of each light curve for each cut applied (see below sections for more information on each cut): 
	- Uncertainty cut: 0x2
	- Chi-square cut: 0x1
	- Control light curve cut: 0x400000
	- Bad day (for averaged light curves): 0x800000

## FILTER: o

### Uncertainty cut
Total percent of data flagged (0x2): 3.84%

### True uncertainties estimation
We can increase the typical uncertainties from 19.00 to 22.15 by adding an additional systematic uncertainty of 11.38 in quadrature
New typical uncertainty is 16.58% greater than old typical uncertainty
Apply true uncertainties estimation set to False
True uncertainties estimation recommended
Skipping procedure...

### Chi-square cut
Chi-square cut 10.00 selected with 0.93% contamination and 0.07% loss
Total percent of data flagged (0x1): 0.13%

### Control light curve cut
Percent of data above x2_max bound (0x100): 11.94%
Percent of data above stn_max bound (0x200): 0.34%
Percent of data above Nclip_max bound (0x400): 0.00%
Percent of data below Ngood_min bound (0x800): 100.00%
Total percent of data flagged as questionable (not masked with control light curve flags but Nclip > 0) (0x80000): 0.00%
Total percent of data flagged as bad (0x400000): 100.00%

### ATLAS template change correction
Corrective flux -4.00 uJy added to first region
Corrective flux -4.00 uJy added to third region
Corrective flux 6.00 uJy added to global region

After the cuts are applied, the light curves are resaved with the new "Mask" column.
Total percent of data flagged as bad (0xc00003): 100.00

### Averaging cleaned light curves
Total percent of binned data flagged (0x800000): 100.00%
The averaged light curves are then saved in a new file with the MJD bin size added to the filename.

## FILTER: c

### Uncertainty cut
Total percent of data flagged (0x2): 2.30%

### True uncertainties estimation
We can increase the typical uncertainties from 12.00 to 13.02 by adding an additional systematic uncertainty of 5.05 in quadrature
New typical uncertainty is 8.50% greater than old typical uncertainty
Apply true uncertainties estimation set to False
True uncertainties estimation not needed; skipping procedure...

### Chi-square cut
Chi-square cut 10.00 selected with 0.49% contamination and 0.06% loss
Total percent of data flagged (0x1): 0.36%

### Control light curve cut
Percent of data above x2_max bound (0x100): 11.74%
Percent of data above stn_max bound (0x200): 0.24%
Percent of data above Nclip_max bound (0x400): 0.00%
Percent of data below Ngood_min bound (0x800): 100.00%
Total percent of data flagged as questionable (not masked with control light curve flags but Nclip > 0) (0x80000): 0.00%
Total percent of data flagged as bad (0x400000): 100.00%

### ATLAS template change correction
Corrective flux -2.50 uJy added to first region
Corrective flux 7.50 uJy added to third region
Corrective flux 4.50 uJy added to global region

After the cuts are applied, the light curves are resaved with the new "Mask" column.
Total percent of data flagged as bad (0xc00003): 100.00

### Averaging cleaned light curves
Total percent of binned data flagged (0x800000): 100.00%
The averaged light curves are then saved in a new file with the MJD bin size added to the filename.