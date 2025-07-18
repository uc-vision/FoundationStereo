# Exports all variables from .env
 set -o allexport && source $PIXI_PROJECT_ROOT/.env && set +o allexport