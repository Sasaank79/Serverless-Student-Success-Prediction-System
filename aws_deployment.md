# AWS Lambda Deployment Guide (EC2 Method)

Since AWS CloudShell has limited storage (1GB), we will use a **Free Tier EC2 Instance** (30GB storage) to build and push our Docker image. This is the standard, robust way to deploy large ML models.

## Prerequisites
*   AWS Account.
*   Terminal on your Mac.
*   Project files ready on your Mac.

---

## Step 1: Launch a Temporary EC2 Instance
1.  Log in to the **AWS Console** and search for **EC2**.
2.  Click **Launch Instance**.
3.  **Name**: `ML-Builder`.
4.  **OS Image**: Choose **Amazon Linux 2023** (Free tier eligible).
5.  **Instance Type**: Choose **t2.micro** (Free tier eligible).
6.  **Key Pair**:
    *   Click **Create new key pair**.
    *   Name: `ml-key`.
    *   Type: `RSA`.
    *   Format: `.pem`.
    *   Click **Create key pair**.
    *   **IMPORTANT**: The `ml-key.pem` file will download to your Mac. **Move it to your project folder**.
7.  **Network Settings**: Ensure "Allow SSH traffic from" is checked (My IP or Anywhere).
8.  **Storage**: Change **8 GiB** to **29 GiB** (Free tier allows up to 30GB).
9.  Click **Launch Instance**.

---

## Step 2: Prepare & Transfer Code
1.  **On your Mac**, open your terminal and go to your project folder:
    ```bash
    cd /Users/mypc/Documents/ML_EtE_Pipeline
    ```
2.  **Set permissions for the key**:
    ```bash
    chmod 400 ml-key.pem
    ```
3.  **Zip your project** (excluding heavy local folders):
    ```bash
    zip -r project.zip . -x "venv/*" ".git/*" "__pycache__/*"
    ```
4.  **Get your EC2 IP Address**:
    *   Go to the AWS Console -> EC2 -> Instances.
    *   Click your `ML-Builder` instance.
    *   Copy the **Public IPv4 address** (e.g., `54.123.45.67`).
5.  **Upload the zip to EC2** (Replace `YOUR_EC2_IP` with the actual IP):
    ```bash
    scp -i ml-key.pem project.zip ec2-user@YOUR_EC2_IP:~
    ```

---

## Step 3: Connect & Setup Docker
1.  **SSH into the instance**:
    ```bash
    ssh -i ml-key.pem ec2-user@YOUR_EC2_IP
    ```
2.  **Install Docker & Git** (Run these commands inside EC2):
    ```bash
    sudo yum update -y
    sudo yum install -y docker git
    sudo service docker start
    sudo usermod -a -G docker ec2-user
    ```
3.  **Refresh permissions** (Log out and log back in):
    *   Type `exit`.
    *   Run the `ssh` command again.

---

## Step 4: Build & Push to ECR
1.  **Unzip the project**:
    ```bash
    unzip project.zip -d ml_project
    cd ml_project
    ```
2.  **Login to ECR**:
    *   Go to AWS Console -> **Elastic Container Registry**.
    *   Click your repository (`student-dropout-api`).
    *   Click **View push commands**.
    *   Copy **Command 1** (Login command) and run it in your EC2 terminal.
3.  **Build the Image**:
    ```bash
    docker build -t student-dropout-api .
    ```
    *(Note: We can use the standard `Dockerfile` or `Dockerfile.lambda` depending on your preference, but usually `Dockerfile.lambda` is best for Lambda).*
4.  **Tag & Push**:
    *   Copy **Command 3** (Tag) from the console and run it.
    *   Copy **Command 4** (Push) from the console and run it.

---

## Step 5: Deploy to Lambda
1.  Go to **AWS Lambda** console.
2.  **Create Function** -> **Container Image**.
3.  Name: `StudentDropoutFunction`.
4.  **Container Image URI**: Click **Browse images** and select the one you just pushed.
5.  **Architecture**: Select **x86_64**.
6.  Click **Create function**.
7.  **Configuration**:
    *   Go to **Configuration** -> **General configuration** -> **Edit**.
    *   **Memory**: Increase to **1024 MB** (or 2048 MB).
    *   **Timeout**: Increase to **30 seconds**.
    *   Click **Save**.

---

## Step 6: CLEANUP (Crucial!)
To avoid any charges:
1.  **Terminate EC2**: Go to EC2 Console -> Select Instance -> Instance State -> **Terminate**.
2.  **Delete ECR Repo**: Go to ECR Console -> Select Repo -> **Delete**.
