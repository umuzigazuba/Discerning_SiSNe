# %%
import subprocess

# %%

# File path to your input text file
input_file = "ZTF_api/ZTF_info.txt"

# Your email and password for the forced photometry request
email = "umuzigazuba@strw.leidenuniv.nl"
userpass = "lvwb246"

# Open the file and process each line
with open(input_file, 'r') as file:
    for line in file:
        # Split the line into components (assumes space-separated values)
        ra_deg, dec_deg, jd_start, jd_end = line.strip().split()
        
        # Construct the wget command
        command = f'wget --http-user=ztffps --http-passwd=dontgocrazy! -O log.txt "https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi?ra={ra_deg}&dec={dec_deg}&jdstart={jd_start}&jdend={jd_end}&email={email}&userpass={userpass}"'
        
        # Execute the wget command for each line in the file
        subprocess.run(command, shell=True)

# %%

