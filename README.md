# Crop-disease-classification

오늘날 반려식물부터 도시 농부, 주말농장까지 주위에서 농작물을 기르는 사람들을 예전보다 더 잘 찾아볼 수 있습니다. 

농촌진흥청은 도시 농업의 총가치가 5조원이 넘는다는 분석을 했습니다.
작년 기준 도시농부는 200만명을 넘어섰고 도시 텃밭 면적은 1052핵타르로 여의도 면적의 3.6배에 달합니다.

하지만 식물은 다양한 외부 환경 요인과 병원균에 의해 질병을 앓게 될 수 있습니다. 
식물에 문제가 생겼을 때, 질병의 원인을 알아내고 적절한 치료법을 찾기는 쉽지 않습니다.
대부분의 사람들은 인터넷 검색을 통해 정보를 찾으려 하지만, 정확한 진단 및 치료법을 얻기가 어렵습니다.

최근 딥러닝 및 인공지능 기술의 발전으로 이미지 기반의 질병 진단이 가능해졌지만, 아직 이러한 기술을 활용하여 사용자가 쉽게 사용할 수 있도록 적용한 플랫폼은 존재하지 않습니다.
따라서 현재 비전문가가 병충해 정보를 알아보기 위해서는 인터넷 검색 또는 사진을 식물 커뮤니티에 올려 질문을 하는 방법뿐입니다. 
이는 정확하지 않은 정보 또는 오랜 시간이 걸린다는 문제가 있습니다.

따라서 본 프로젝트는 이미지 분석 기술을 사용한 식물 질병 진단 을 제공하는 것을 목표로 합니다.
이를 통해 식물 애호가들은 더 건강한 식물을 키우는 데 도움을 받을 수 있을 것입니다.

## 1. Dataset

![added](https://github.com/mukkbo/Crop-disease-classification/assets/133736337/7436df31-ccfc-48fe-92fc-48e97c00ec54)

### 포도 질병 예시

<div align="center">
  <img src="https://github.com/mukkbo/Crop-disease-classification/assets/133736337/62b15472-c531-40f7-9a65-7dde9bd3632a" width="224" height="224"/>
  <p align="center">노균병</p>
</div>
<div align="center">
  <img src="https://github.com/mukkbo/Crop-disease-classification/assets/133736337/3989ef26-0e54-406f-b659-89e850137f24" width="224" height="224"/>
   <p align="center">탄저병</p>
</div>
<div align="center">
  <img src="https://github.com/mukkbo/Crop-disease-classification/assets/133736337/ff0773d1-b6ec-4c1d-80d7-1454a75733f7" width="224" height="224"/>
   <p align="center">축과병</p>
</div align="center">
<div align="center">
  <img src="https://github.com/mukkbo/Crop-disease-classification/assets/133736337/e38f617d-852d-44bd-9056-bce1c54d26bc"  width="224" height="224"/>
   <p align="center">일소피해</p>
</div>

## 2. Model
### a. ResNet50
### b. Simple-ViT
### c. MAE-ViT
