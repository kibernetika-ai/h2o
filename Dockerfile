FROM kuberlab/mlbase:cpu-36-full

RUN \
  echo 'DPkg::Post-Invoke {"/bin/rm -f /var/cache/apt/archives/*.deb || true";};' | tee /etc/apt/apt.conf.d/no-cache && \
  echo "deb http://mirror.math.princeton.edu/pub/ubuntu xenial main universe" >> /etc/apt/sources.list && \
  apt-get update -q -y && \
  apt-get dist-upgrade -y && \
  apt-get clean && \
  rm -rf /var/cache/apt/* && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y wget unzip openjdk-8-jdk && \
  apt-get clean && \
# Fetch h2o latest_stable
  wget http://h2o-release.s3.amazonaws.com/h2o/rel-yau/3/h2o-3.26.0.3.zip -O /opt/h2o.zip && \
  unzip -d /opt /opt/h2o.zip && \
  rm /opt/h2o.zip && \
  cd /opt && \
  cd `find . -name 'h2o.jar' | sed 's/.\///;s/\/h2o.jar//g'` && \
  cp h2o.jar /opt && \
  pip install `find . -name "*.whl"` && \
  cd / && \
#  wget https://raw.githubusercontent.com/h2oai/h2o-3/master/docker/start-h2o-docker.sh && \
#  chmod +x start-h2o-docker.sh && \
# Get Content
  wget http://s3.amazonaws.com/h2o-training/mnist/train.csv.gz && \
  gunzip train.csv.gz && \
  wget https://raw.githubusercontent.com/laurendiperna/Churn_Scripts/master/Extraction_Script.py  && \
  wget https://raw.githubusercontent.com/laurendiperna/Churn_Scripts/master/Transformation_Script.py && \
  wget https://raw.githubusercontent.com/laurendiperna/Churn_Scripts/master/Modeling_Script.py

COPY ./docker-entrypoint.sh /
# Define a mountable data directory
#VOLUME \
#  ["/data"]

# Define the working directory
#WORKDIR \
#  /data

EXPOSE 54321
EXPOSE 54322

#ENTRYPOINT ["java", "-Xmx4g", "-jar", "/opt/h2o.jar"]
ENTRYPOINT ["/docker-entrypoint.sh"]
