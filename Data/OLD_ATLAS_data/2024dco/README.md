# SN 2024dco Light Curve Cleaning and Averaging

The ATLAS SN light curves are separated by filter (orange and cyan) and labelled as such in the file name. Averaged light curves contain an additional number in the file name that represents the MJD bin size used. Control light curves are located in the "controls" subdirectory and follow the same naming scheme, only with their control index added after the SN name.

The following details the file names for each of the light curve versions:
	- SN light curves: 2024dco.o.lc.txt and 2024dco.c.lc.txt
	- Averaged light curves: 2024dco.o.1.00days.lc.txt and 2024dco.c.1.00days.lc.txt
	- Control light curves, where X=001,...,002: 2024dco_iX.o.lc.txt and 2024dco_iX.c.lc.txt

The following summarizes the hex values in the "Mask" column of each light curve for each cut applied (see below sections for more information on each cut): 
	- Uncertainty cut: 0x2
	- Chi-square cut: 0x1
	- Control light curve cut: 0x400000
	- Bad day (for averaged light curves): 0x800000

## FILTER: o

### Uncertainty cut
Total percent of data flagged (0x2): 4.03%

### True uncertainties estimation
We can increase the typical uncertainties from 18.00 to 20.08 by adding an additional systematic uncertainty of 8.90 in quadrature
New typical uncertainty is 11.55% greater than old typical uncertainty
Apply true uncertainties estimation set to False
True uncertainties estimation recommended
Skipping procedure...

### Chi-square cut
Chi-square cut 10.00 selected with 0.71% contamination and 0.05% loss
Total percent of data flagged (0x1): 0.10%

### Control light curve cut
Percent of data above x2_max bound (0x100): 12.19%
Percent of data above stn_max bound (0x200): 0.45%
Percent of data above Nclip_max bound (0x400): 0.00%
Percent of data below Ngood_min bound (0x800): 100.00%
Total percent of data flagged as questionable (not masked with control light curve flags but Nclip > 0) (0x80000): 0.00%
Total percent of data flagged as bad (0x400000): 100.00%

### ATLAS template change correction
Corrective flux -1.00 uJy added to first region
Corrective flux 3.50 uJy added to third region
Corrective flux 14.00 uJy added to global region

After the cuts are applied, the light curves are resaved with the new "Mask" column.
Total percent of data flagged as bad (0xc00003): 100.00

### Averaging cleaned light curves
Total percent of binned data flagged (0x800000): 100.00%
The averaged light curves are then saved in a new file with the MJD bin size added to the filename.

## FILTER: c

### Uncertainty cut
Total percent of data flagged (0x2): 1.25%

### True uncertainties estimation
We can increase the typical uncertainties from 10.00 to 11.86 by adding an additional systematic uncertainty of 6.38 in quadrature
New typical uncertainty is 18.61% greater than old typical uncertainty
Apply true uncertainties estimation set to False
True uncertainties estimation recommended
Skipping procedure...

### Chi-square cut
Chi-square cut 10.00 selected with 0.69% contamination and 0.00% loss
Total percent of data flagged (0x1): 0.13%

### Control light curve cut
Percent of data above x2_max bound (0x100): 14.02%
Percent of data above stn_max bound (0x200): 0.38%
Percent of data above Nclip_max bound (0x400): 0.00%
Percent of data below Ngood_min bound (0x800): 100.00%
Total percent of data flagged as questionable (not masked with control light curve flags but Nclip > 0) (0x80000): 0.00%
Total percent of data flagged as bad (0x400000): 100.00%

### ATLAS template change correction
Corrective flux -5.00 uJy added to first region
Corrective flux 2.00 uJy added to third region
Corrective flux 8.50 uJy added to global region

After the cuts are applied, the light curves are resaved with the new "Mask" column.
Total percent of data flagged as bad (0xc00003): 100.00

### Averaging cleaned light curves
Total percent of binned data flagged (0x800000): 100.00%
The averaged light curves are then saved in a new file with the MJD bin size added to the filename.