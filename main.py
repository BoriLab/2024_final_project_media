import os
from final_main import get_data
from mask_Seg import mask_img
from beauty_gan import beauty

def main():
    # Step 1: 실행할 이미지 경로
    input_image_path = "./input_img.jpg"
    mask_output_path = "./img/reference2.jpg"  # MaskSeg에서 출력될 이미지 경로
    final_output_dir = "./img"  # 최종 결과 출력 디렉토리

    lab_value = get_data(input_image_path, use_db=False)
    lab_value
    #본래는 화장품 데이터 set이랑 비교해서 피부색에 알맞는 화장품 색상을 넣어야 하지만 
    #코드 실행을 확인해야 함으로 임시로 아무 화장품색으로 고정해서 제출했습니다.
    mask_img(input_image_path,lab_value[0],lab_value[1],lab_value[2])
    beauty(input_image_path, mask_output_path)
    
if __name__ == "__main__":
    main()
