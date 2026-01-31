import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import json


class LeafDiseaseAnalyzer:
    def __init__(self, pixels_per_cm: float = 62.0, min_leaf_area_cm: float = 0.5):
        self.pixels_per_cm = pixels_per_cm
        self.min_leaf_area_cm = min_leaf_area_cm

        self.color_ranges = {
            'healthy_green': {'lower': [5, 30, 60], 'upper': [100, 255, 255]},
            'light_green': {'lower': [5, 30, 60], 'upper': [100, 255, 255]},
            'yellow_damage': {'lower': [0, 70, 50], 'upper': [15, 255, 255]},
            'brown_damage': {'lower': [130, 50, 50], 'upper': [180, 255, 255]},
            'white_damage': {'lower': [0, 0, 200], 'upper': [180, 30, 255]}
        }

        self.results = {
            'total_leaves': 0,
            'leaves': [],
            'overall_disease_percentage': 0.0,
            'pixels_per_cm': pixels_per_cm,
            'filtered_out_leaves': 0,
            'min_leaf_area_cm': min_leaf_area_cm
        }

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        img_processed = cv2.medianBlur(img_processed, 3)

        hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)
        return img, hsv

    def create_color_mask(self, hsv: np.ndarray, color_name: str) -> np.ndarray:
        lower = np.array(self.color_ranges[color_name]['lower'])
        upper = np.array(self.color_ranges[color_name]['upper'])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.medianBlur(mask, 3)

        return mask

    def detect_individual_leaves(self, green_mask: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leaf_info = []
        filtered_count = 0

        for contour in contours:
            area_px = cv2.contourArea(contour)
            area_cm = area_px / (self.pixels_per_cm ** 2)

            if area_cm < self.min_leaf_area_cm:
                filtered_count += 1
                print(f"Лист отфильтрован: площадь {area_cm:.2f} см² < {self.min_leaf_area_cm} см²")
                continue

            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) >= 3:
                leaf_info.append((approx, area_px, area_cm))
            else:
                filtered_count += 1

        leaf_info.sort(key=lambda x: x[1], reverse=True)
        print(f"Обнаружено {len(leaf_info)} листьев (отфильтровано {filtered_count})")

        return leaf_info

    def calculate_leaf_metrics(self, contour: np.ndarray,
                               contour_area_px: float,
                               contour_area_cm: float,
                               green_mask: np.ndarray,
                               damage_mask: np.ndarray) -> Dict:
        leaf_mask = np.zeros_like(green_mask)
        cv2.drawContours(leaf_mask, [contour], -1, 255, -1)

        if cv2.countNonZero(leaf_mask) == 0:
            print("Предупреждение: маска листа пустая!")
            return None

        green_in_leaf = cv2.bitwise_and(green_mask, leaf_mask)
        green_area_px = cv2.countNonZero(green_in_leaf)
        green_area_cm = green_area_px / (self.pixels_per_cm ** 2)

        damage_in_leaf = cv2.bitwise_and(damage_mask, leaf_mask)
        damage_area_px = cv2.countNonZero(damage_in_leaf)
        damage_area_cm = damage_area_px / (self.pixels_per_cm ** 2)

        green_without_damage = cv2.bitwise_and(green_in_leaf, cv2.bitwise_not(damage_in_leaf))
        green_area_px_corrected = cv2.countNonZero(green_without_damage)
        green_area_cm_corrected = green_area_px_corrected / (self.pixels_per_cm ** 2)

        total_leaf_area_px = contour_area_px
        total_leaf_area_cm = contour_area_cm

        if total_leaf_area_px > 0:
            disease_percentage_total = (damage_area_px / total_leaf_area_px) * 100
        else:
            disease_percentage_total = 0

        sum_areas = green_area_px_corrected
        area_difference = abs(total_leaf_area_px - sum_areas)
        area_difference_percent = (area_difference / total_leaf_area_px * 100) if total_leaf_area_px > 0 else 0


        return {
            'contour_area_px': total_leaf_area_px,
            'contour_area_cm': total_leaf_area_cm,

            'healthy_green_area_px': green_area_px_corrected,
            'healthy_green_area_cm': green_area_cm_corrected,

            'damage_area_px': damage_area_px,
            'damage_area_cm': damage_area_cm,

            'disease_percentage_by_damage': disease_percentage_total,

            'area_check_difference_percent': area_difference_percent,
            'damage_mask': damage_in_leaf,
            'healthy_mask': green_without_damage,
            'leaf_mask': leaf_mask
        }

    def analyze_image(self, image_path: str, use_convex_hull: bool = False) -> Dict:
        img, hsv = self.preprocess_image(image_path)

        try:
            self.find_ruler(img)
        except:
            print("Поиск линейки пропущен, используется стандартный коэффициент")

        print("Создание цветовых масок...")

        green_mask = self.create_color_mask(hsv, 'healthy_green')
        light_green_mask = self.create_color_mask(hsv, 'light_green')
        green_mask = cv2.bitwise_or(green_mask, light_green_mask)
        yellow_damage = self.create_color_mask(hsv, 'yellow_damage')
        brown_damage = self.create_color_mask(hsv, 'brown_damage')
        white_damage = self.create_color_mask(hsv, 'white_damage')

        damage_mask = cv2.bitwise_or(yellow_damage, brown_damage)
        damage_mask = cv2.bitwise_or(damage_mask, white_damage)

        leaf_info = self.detect_individual_leaves(green_mask)

        if not leaf_info:
            print("Листья не обнаружены")
            return self.results

        results_list = []
        all_damage_mask = np.zeros_like(damage_mask)

        print("\nАнализ обнаруженных листьев:")
        for i, (contour, area_px, area_cm) in enumerate(leaf_info):
            print(f"\nЛист {i + 1}: начальная площадь {area_cm:.2f} см²")

            if use_convex_hull:
                contour = cv2.convexHull(contour)
                area_px = cv2.contourArea(contour)
                area_cm = area_px / (self.pixels_per_cm ** 2)
                print(f"  Выпуклая оболочка: площадь {area_cm:.2f} см²")

            metrics = self.calculate_leaf_metrics(
                contour=contour,
                contour_area_px=area_px,
                contour_area_cm=area_cm,
                green_mask=green_mask,
                damage_mask=damage_mask
            )

            if metrics is None:
                print(f"  Лист {i + 1} пропущен из-за ошибки расчета")
                continue

            if metrics['contour_area_cm'] < self.min_leaf_area_cm:
                print(
                    f"  Лист {i + 1} пропущен: уточненная площадь {metrics['contour_area_cm']:.2f} см² меньше минимальной")
                continue

            results_list.append(metrics)

            all_damage_mask = cv2.bitwise_or(all_damage_mask, metrics['damage_mask'])

            print(f"  Результат: площадь={metrics['contour_area_cm']:.2f} см², "
                  f"зеленая={metrics['healthy_green_area_cm']:.2f} см², "
                  f"повреждения={metrics['damage_area_cm']:.2f} см², "
                  f"процент повреждений={metrics['disease_percentage_by_damage']:.1f}%")

        visualization = self.create_visualization(
            img,
            [info[0] for info in leaf_info if len(leaf_info) > 0],
            results_list,
            all_damage_mask
        )

        if results_list:
            total_damage_percentage = np.mean([r['disease_percentage_by_damage'] for r in results_list])
        else:
            total_damage_percentage = 0

        self.results = {
            'total_leaves': len(results_list),
            'leaves': results_list,
            'overall_disease_percentage_damage': total_damage_percentage,
            'total_healthy_area_cm': sum([r['healthy_green_area_cm'] for r in results_list]),
            'total_damage_area_cm': sum([r['damage_area_cm'] for r in results_list]),
            'total_contour_area_cm': sum([r['contour_area_cm'] for r in results_list]),
            'pixels_per_cm': self.pixels_per_cm,
            'min_leaf_area_cm': self.min_leaf_area_cm,
            'filtered_out_leaves': len(leaf_info) - len(results_list),
            'visualization': visualization
        }

        return self.results

    def create_visualization(self, img: np.ndarray, leaf_contours: List[np.ndarray],
                             results_list: List[Dict], damage_mask: np.ndarray) -> np.ndarray:
        vis_img = img.copy()

        damage_overlay = vis_img.copy()
        damage_overlay[damage_mask > 0] = (0, 0, 255)
        cv2.addWeighted(damage_overlay, 0.3, vis_img, 0.7, 0, vis_img)

        colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0)]

        for i, (contour, result) in enumerate(zip(leaf_contours[:len(results_list)], results_list)):
            color = colors[i % len(colors)]

            cv2.drawContours(vis_img, [contour], -1, color, 2)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                text1 = f"L{i + 1}: {result['contour_area_cm']:.1f}cm²"
                text2 = f"Dmg: {result['disease_percentage_by_damage']:.1f}%"

                cv2.putText(vis_img, text1, (cx - 80, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(vis_img, text2, (cx - 80, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if results_list:
            cv2.putText(vis_img,
                        f"Leaves: {len(results_list)}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis_img



def main():
    analyzer = LeafDiseaseAnalyzer(pixels_per_cm=62.0, min_leaf_area_cm=10)

    image_path = "7.jpg"

    try:
        results = analyzer.analyze_image(image_path, use_convex_hull=False)

        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА (КОРРЕКТНЫЙ РАСЧЕТ):")
        print("=" * 60)

        print(f"\nОБЩАЯ СТАТИСТИКА:")
        print(f"  Всего листьев: {results['total_leaves']}")
        print(f"  Отфильтровано: {results['filtered_out_leaves']}")
        print(f"  Минимальная площадь: {analyzer.min_leaf_area_cm} см²")
        print(f"  Пикселей/см: {results['pixels_per_cm']:.2f}")

        print(f"\nОБЩАЯ ПЛОЩАДЬ:")
        print(f"  Общая площадь листьев: {results['total_contour_area_cm']:.1f} см²")
        print(f"  Здоровая площадь: {results['total_healthy_area_cm']:.1f} см²")
        print(f"  Площадь повреждений: {results['total_damage_area_cm']:.1f} см²")


        print(f"\nПРОЦЕНТ ПОРАЖЕНИЯ:")
        print(f"  По повреждениям: {results['overall_disease_percentage_damage']:.1f}%")


        if results['leaves']:
            print(f"\nДЕТАЛИ ПО ЛИСТЬЯМ:")
            for i, leaf in enumerate(results['leaves']):
                print(f"  Лист {i + 1}:")
                print(f"    Площадь: {leaf['contour_area_cm']:.2f} см²")
                print(
                    f"    Здоровая: {leaf['healthy_green_area_cm']:.2f} см² ({leaf['healthy_green_area_cm'] / leaf['contour_area_cm'] * 100:.1f}%)")
                print(
                    f"    Повреждения: {leaf['damage_area_cm']:.2f} см² ({leaf['disease_percentage_by_damage']:.1f}%)")
                print(f"    Проверка: разница {leaf['area_check_difference_percent']:.1f}%")


        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title('Оригинальное изображение')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        vis_rgb = cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB)
        plt.imshow(vis_rgb)
        plt.title(f'Результат анализа ({results["total_leaves"]} листьев)')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        if results['leaves']:
            leaf_nums = range(1, len(results['leaves']) + 1)
            damage_percents = [leaf['disease_percentage_by_damage'] for leaf in results['leaves']]

            width = 0.35
            plt.bar([x - width / 2 for x in leaf_nums], damage_percents, width, label='По повреждениям', color='red')

            plt.xlabel('Номер листа')
            plt.ylabel('Процент поражения (%)')
            plt.title('Процент поражения листьев')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
            plt.axis('off')

        plt.subplot(2, 2, 4)
        if results['leaves']:
            leaf_nums = range(1, len(results['leaves']) + 1)
            healthy_areas = [leaf['healthy_green_area_cm'] for leaf in results['leaves']]
            damage_areas = [leaf['damage_area_cm'] for leaf in results['leaves']]

            plt.bar(leaf_nums, healthy_areas, label='Здоровая площадь', color='green')
            plt.bar(leaf_nums, damage_areas, bottom=healthy_areas, label='Поврежденная площадь', color='red')

            plt.xlabel('Номер листа')
            plt.ylabel('Площадь (см²)')
            plt.title('Распределение площадей')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('leaf_analysis_corrected.png', dpi=150, bbox_inches='tight')
        plt.savefig('res_v0.3 3.png')
        plt.show()

        analyzer.save_results("leaf_analysis_results_corrected.json")

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()