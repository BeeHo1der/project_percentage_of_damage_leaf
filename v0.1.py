import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import json


class LeafDiseaseAnalyzer:
    def __init__(self, pixels_per_cm: float = 62.0):
        self.pixels_per_cm = pixels_per_cm
        self.color_ranges = {
            'green': {'lower': [25, 30, 30], 'upper': [90, 255, 255]},
            'damage_yellow': {'lower': [5, 50, 20], 'upper': [35, 255, 200]},
            'damage_brown': {'lower': [0, 50, 20], 'upper': [20, 255, 150]},
            'white': {'lower': [0, 0, 200], 'upper': [180, 30, 255]}
        }

        self.results = {
            'total_leaves': 0,
            'leaves': [],
            'overall_disease_percentage': 0.0,
            'pixels_per_cm': pixels_per_cm
        }

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)
        return img, hsv

    def create_color_mask(self, hsv: np.ndarray, color_name: str) -> np.ndarray:
        lower = np.array(self.color_ranges[color_name]['lower'])
        upper = np.array(self.color_ranges[color_name]['upper'])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def find_ruler(self, img: np.ndarray) -> Optional[float]:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        if lines is not None:

            ruler_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                if length > 200 and abs(angle) < 10:
                    ruler_lines.append(line[0])

            if len(ruler_lines) > 0:
                longest_line = max(ruler_lines,
                                   key=lambda x: np.sqrt((x[2] - x[0]) ** 2 + (x[3] - x[1]) ** 2))
                line_length_px = np.sqrt((longest_line[2] - longest_line[0]) ** 2 +
                                         (longest_line[3] - longest_line[1]) ** 2)

                self.pixels_per_cm = line_length_px / 10
                print(f"Найдена линейка. Коэффициент: {self.pixels_per_cm:.2f} пикселей/см")
                return self.pixels_per_cm

        print("Линейка не найдена, используется стандартный коэффициент")
        return None

    def detect_individual_leaves(self, green_mask: np.ndarray, min_area: int = 500) -> List[np.ndarray]:

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leaf_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                leaf_contours.append(approx)

        leaf_contours.sort(key=cv2.contourArea, reverse=True)
        return leaf_contours

    def calculate_leaf_metrics(self, leaf_mask: np.ndarray,
                               green_mask: np.ndarray,
                               damage_mask: np.ndarray) -> Dict:
        green_on_leaf = cv2.bitwise_and(green_mask, leaf_mask)
        green_area_px = cv2.countNonZero(green_on_leaf)
        green_area_cm = green_area_px / (self.pixels_per_cm ** 2)

        damage_on_leaf = cv2.bitwise_and(damage_mask, leaf_mask)
        damage_area_px = cv2.countNonZero(damage_on_leaf)
        damage_area_cm = damage_area_px / (self.pixels_per_cm ** 2)

        total_relevant_area_px = green_area_px + damage_area_px

        if total_relevant_area_px > 0:
            disease_percentage = (damage_area_px / total_relevant_area_px) * 100
        else:
            disease_percentage = 0.0

        total_contour_area_px = cv2.countNonZero(leaf_mask)
        total_contour_area_cm = total_contour_area_px / (self.pixels_per_cm ** 2)

        return {
            'total_contour_area_px': total_contour_area_px,
            'total_contour_area_cm': total_contour_area_cm,
            'green_area_px': green_area_px,
            'green_area_cm': green_area_cm,
            'damage_area_px': damage_area_px,
            'damage_area_cm': damage_area_cm,
            'total_relevant_area_px': total_relevant_area_px,
            'total_relevant_area_cm': total_relevant_area_px / (self.pixels_per_cm ** 2),
            'disease_percentage': disease_percentage,
            'damage_mask': damage_on_leaf,
            'green_mask': green_on_leaf
        }

    def create_visualization(self, img: np.ndarray, leaf_contours: List[np.ndarray],
                             results_list: List[Dict], damage_mask: np.ndarray) -> np.ndarray:
        vis_img = img.copy()

        overlay = img.copy()
        alpha = 0.3  # Прозрачность

        overlay[damage_mask > 0] = (0, 0, 255)
        cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

        colors = [
            (0, 255, 0),  # Зеленый
            (255, 255, 0),  # Голубой
            (255, 0, 255),  # Розовый
            (0, 255, 255),  # Желтый
            (255, 0, 0)  # Синий
        ]

        for i, (contour, result) in enumerate(zip(leaf_contours, results_list)):
            color = colors[i % len(colors)]

            cv2.drawContours(vis_img, [contour], -1, color, 2)

            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                text = f"Лист {i + 1}: {result['disease_percentage']:.1f}%"
                cv2.putText(vis_img, text, (cx - 100, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                area_text = f"Здоров: {result['green_area_cm']:.1f} см², Поврежд: {result['damage_area_cm']:.1f} см²"
                cv2.putText(vis_img, area_text, (cx - 150, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if results_list:
            total_percentage = np.mean([r['disease_percentage'] for r in results_list])
            total_green = sum([r['green_area_cm'] for r in results_list])
            total_damage = sum([r['damage_area_cm'] for r in results_list])

            stats_text = f"Листьев: {len(leaf_contours)} | Средний % повреждения: {total_percentage:.1f}%"
            area_text = f"Общая здор. площадь: {total_green:.1f} см² | Общая поврежд. площадь: {total_damage:.1f} см²"

            cv2.putText(vis_img, stats_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis_img, area_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return vis_img

    def analyze_image(self, image_path: str, use_convex_hull: bool = False) -> Dict:
        img, hsv = self.preprocess_image(image_path)

        try:
            self.find_ruler(img)
        except:
            print("Поиск линейки пропущен, используется стандартный коэффициент")

        # 3. Создание масок
        green_mask = self.create_color_mask(hsv, 'green')
        yellow_damage = self.create_color_mask(hsv, 'damage_yellow')
        brown_damage = self.create_color_mask(hsv, 'damage_brown')
        white_damage = self.create_color_mask(hsv, 'white')

        damage_mask = cv2.bitwise_or(yellow_damage, brown_damage)
        damage_mask = cv2.bitwise_or(damage_mask, white_damage)

        leaf_contours = self.detect_individual_leaves(green_mask)

        if not leaf_contours:
            print("Листья не обнаружены")
            return self.results

        results_list = []
        all_damage_mask = np.zeros_like(damage_mask)

        for i, contour in enumerate(leaf_contours):
            leaf_mask = np.zeros_like(green_mask)
            cv2.drawContours(leaf_mask, [contour], -1, 255, -1)

            if use_convex_hull:
                hull = cv2.convexHull(contour)
                leaf_mask = np.zeros_like(green_mask)
                cv2.drawContours(leaf_mask, [hull], -1, 255, -1)

            metrics = self.calculate_leaf_metrics(leaf_mask, green_mask, damage_mask)
            results_list.append(metrics)

            all_damage_mask = cv2.bitwise_or(all_damage_mask, metrics['damage_mask'])

            print(f"Лист {i + 1}: зеленая площадь={metrics['green_area_cm']:.1f} см², "
                  f"поврежденная площадь={metrics['damage_area_cm']:.1f} см², "
                  f"повреждение={metrics['disease_percentage']:.1f}%")

        visualization = self.create_visualization(img, leaf_contours, results_list, all_damage_mask)

        self.results = {
            'total_leaves': len(leaf_contours),
            'leaves': results_list,
            'overall_disease_percentage': np.mean([r['disease_percentage'] for r in results_list]),
            'total_green_area_cm': sum([r['green_area_cm'] for r in results_list]),
            'total_damage_area_cm': sum([r['damage_area_cm'] for r in results_list]),
            'pixels_per_cm': self.pixels_per_cm,
            'visualization': visualization
        }

        return self.results

    def plot_results(self, image_path: str):
        results = self.analyze_image(image_path)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Анализ болезней листьев', fontsize=16)

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, hsv = self.preprocess_image(image_path)
        green_mask = self.create_color_mask(hsv, 'green')
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        damage_mask = cv2.bitwise_or(mask1, mask2)

        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Оригинальное изображение')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(green_mask, cmap='gray')
        axes[0, 1].set_title('Маска здоровых областей')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(damage_mask, cmap='hot')
        axes[0, 2].set_title('Маска повреждений')
        axes[0, 2].axis('off')

        vis_rgb = cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(vis_rgb)
        axes[1, 0].set_title('Результат анализа')
        axes[1, 0].axis('off')

        leaf_numbers = range(1, len(results['leaves']) + 1)
        percentages = [leaf['disease_percentage'] for leaf in results['leaves']]

        axes[1, 1].bar(leaf_numbers, percentages,
                       color=['green' if p < 20 else 'orange' if p < 50 else 'red' for p in percentages])
        axes[1, 1].set_title('Процент повреждения по листьям')
        axes[1, 1].set_xlabel('Номер листа')
        axes[1, 1].set_ylabel('Процент повреждения (%)')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 1].axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Низкий (<20%)')
        axes[1, 1].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Средний (<50%)')
        axes[1, 1].legend(fontsize=8)

        leaf_numbers = range(1, len(results['leaves']) + 1)
        green_areas = [leaf['green_area_cm'] for leaf in results['leaves']]
        damage_areas = [leaf['damage_area_cm'] for leaf in results['leaves']]

        axes[1, 2].bar(leaf_numbers, green_areas, color='green', label='Здоровая площадь')
        axes[1, 2].bar(leaf_numbers, damage_areas, bottom=green_areas, color='red', label='Поврежденная площадь')
        axes[1, 2].set_title('Площадь листьев (зеленая и поврежденная)')
        axes[1, 2].set_xlabel('Номер листа')
        axes[1, 2].set_ylabel('Площадь (см²)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('my_plot.png')
        plt.show()

    def save_results(self, output_path: str):
        json_results = self.results.copy()
        if 'visualization' in json_results:
            del json_results['visualization']

        for leaf in json_results['leaves']:
            if 'damage_mask' in leaf:
                del leaf['damage_mask']
            if 'green_mask' in leaf:
                del leaf['green_mask']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        print(f"Результаты сохранены в {output_path}")


# Пример использования
def main():

    analyzer = LeafDiseaseAnalyzer(pixels_per_cm=62.0)


    image_path = "1.png"

    try:
        analyzer.plot_results(image_path)

        results = analyzer.results
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("=" * 60)
        print(f"Всего листьев: {results['total_leaves']}")
        print(f"Средний процент повреждения: {results['overall_disease_percentage']:.1f}%")
        print(f"Общая здоровая площадь: {results['total_green_area_cm']:.1f} см²")
        print(f"Общая поврежденная площадь: {results['total_damage_area_cm']:.1f} см²")
        print(f"Коэффициент пиксель/см: {results['pixels_per_cm']:.2f}")

        print("\nДетали по листьям:")
        for i, leaf in enumerate(results['leaves']):
            print(f"  Лист {i + 1}: {leaf['disease_percentage']:.1f}% повреждения "
                  f"(здоровая: {leaf['green_area_cm']:.1f} см², "
                  f"поврежденная: {leaf['damage_area_cm']:.1f} см²)")

        analyzer.save_results("leaf_analysis_results.json")

        print("\nТекущие параметры масок цветов:")
        for color, ranges in analyzer.color_ranges.items():
            print(f"{color}: lower={ranges['lower']}, upper={ranges['upper']}")

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()