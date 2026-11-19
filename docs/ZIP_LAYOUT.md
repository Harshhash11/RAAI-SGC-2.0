# ZIP packaging / path resolution

The released archive is deliberately **flat**: it starts with `start.sh`, `requirements.txt`, etc., not `RAAI_SGC_2_COMPLETE_PROJECT/` or `RAAI_SGC_2_FINAL_AUDITED_MAIN_PROJECT/`.

Extract with `mkdir -p ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT && unzip -o RAAI_SGC_2_COMPLETE_PROJECT_READY.zip -d ~/Downloads/RAAI_SGC_2_COMPLETE_PROJECT`.

Within that directory, run `./start.sh setup` and `./start.sh chat`.

The project SHA256 manifest covers source artifacts, excluding the manifest files themselves. It is a package-integrity aid, not proof of research quality.
