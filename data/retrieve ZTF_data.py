# %%
import subprocess
import math
import requests

# %%

### Request data 

# # File path to your input text file
# input_file = "ZTF_api/ZTF_info.txt"

# # Your email and password for the forced photometry request
# email = "umuzigazuba@strw.leidenuniv.nl"
# userpass = "lvwb246"

# # Open the file and process each line
# with open(input_file, 'r') as file:
#     for line in file:
        
#         # Split the line into components (assumes space-separated values)
#         name, ra_deg, dec_deg, jd_start, jd_end = line.strip().split()
        
#         if name != "nan":
#             # Construct the wget command
#             command = f'wget --http-user=ztffps --http-passwd=dontgocrazy! -O log.txt "https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi?ra={ra_deg}&dec={dec_deg}&jdstart={jd_start}&jdend={jd_end}&email={email}&userpass={userpass}"'
            
#             # Execute the wget command for each line in the file
#             subprocess.run(command, shell=True)

# %%

### Download data 

input_file = "ZTF_api/ZTF_info.txt"
output_file = open(f'Requested_Query_ZTF_ForcedPhotometry_Service.txt', 'r')


for line in output_file:

    if line[0]!='r': # avoid header
        line = line.split('\t')
        path = line[10].strip()
        url = f'https://ztfweb.ipac.caltech.edu{path}'

        print(url)

        ra = float(line[1])
        dec = float(line[2])
        name = ""

        with open(input_file, 'r') as file:
            for line in file:

                name_input, ra_deg, dec_deg, _, _ = line.strip().split()

                # searching for name
                tol=1e-9
                if math.isclose(ra, float(ra_deg)) and math.isclose(dec, float(dec_deg)):
                    name = name_input
                    break

        # retrieve photometry data and write in computer
        with requests.Session() as s:
            textdata = s.get(url, auth=('ztffps', 'dontgocrazy!')).text
            with open(f'Data/ZTF_forced_photometry_data/{name}.txt', 'w') as f:
                for line in textdata:
                    f.write(line)


# %%
