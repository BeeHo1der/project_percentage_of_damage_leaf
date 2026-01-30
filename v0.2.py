import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import json


class LeafDiseaseAnalyzer:
    def __init__(self, pixels_per_cm: float = 62.0):
        self.pixels_per_cm = pixels_per_cm
        self.color_ranges = {
            'green': {'lower': [20, 30, 30], 'upper': [130, 255, 255]},
            # 'damage_yellow': {'lower': [5, 50, 20], 'upper': [35, 255, 200]},
            'damage_brown': {'lower': [10, 100, 30], 'upper': [20, 255, 200]}
            # 'white': {'lower': [0, 0, 200], 'upper': [180, 30, 255]}
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
        print("Поиск линейки пропущен")
        return None

    def detect_individual_leaves(self, green_mask: np.ndarray, min_area: int = 500) -> List[np.ndarray]:
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        leaf_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Фильтрация по минимальной площади
            if area > min_area:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                leaf_contours.append(approx)

        leaf_contours.sort(key=cv2.contourArea, reverse=True)
        print(f"Найдено {len(leaf_contours)} листьев (фильтрация: min_area={min_area} px)")
        return leaf_contours

    def calculate_leaf_metrics(self, leaf_mask: np.ndarray,
                               green_mask: np.ndarray,
                               damage_mask: np.ndarray) -> Dict:
        """Расчет метрик для одного листа"""
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
        alpha = 0.3
        overlay[damage_mask > 0] = (0, 0, 255)
        cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

        # Цвета для разных листьев
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

                text = f"{i + 1}: {result['disease_percentage']:.1f}%"
                cv2.putText(vis_img, text, (cx - 50, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return vis_img

    def analyze_image(self, image_path: str, use_convex_hull: bool = False, min_area: int = 500) -> Dict:
        img, hsv = self.preprocess_image(image_path)

        green_mask = self.create_color_mask(hsv, 'green')
        # yellow_damage = self.create_color_mask(hsv, 'damage_yellow')
        damage_mask = self.create_color_mask(hsv, 'damage_brown')
        # white_damage = self.create_color_mask(hsv, 'white')

        # damage_mask = cv2.bitwise_or(yellow_damage, brown_damage)
        # damage_mask = cv2.bitwise_or(damage_mask, white_damage)

        leaf_contours = self.detect_individual_leaves(green_mask, min_area)

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

            print(f"Лист {i + 1}: {metrics['green_area_cm']:.1f} см² здоровых, "
                  f"{metrics['damage_area_cm']:.1f} см² поврежденных, "
                  f"{metrics['disease_percentage']:.1f}% повреждения")

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

    def plot_results(self, image_path: str, min_area: int = 500):

        results = self.analyze_image(image_path, min_area=min_area)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Анализ болезней листьев', fontsize=16)

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, hsv = self.preprocess_image(image_path)
        green_mask = self.create_color_mask(hsv, 'green')

        # yellow_damage = self.create_color_mask(hsv, 'damage_yellow')
        damage_mask = self.create_color_mask(hsv, 'damage_brown')
        # white_damage = self.create_color_mask(hsv, 'white')
        # damage_mask = cv2.bitwise_or(yellow_damage, brown_damage)
        # damage_mask = cv2.bitwise_or(damage_mask, white_damage)

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

        if results['leaves']:
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

        if results['leaves']:
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
        plt.savefig('leaf_analysis_results.png', dpi=300)
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


def main():
    analyzer = LeafDiseaseAnalyzer(pixels_per_cm=62.0)

    image_path = "1.png"

    min_area = 2000

    try:
        analyzer.plot_results(image_path, min_area=min_area)

        results = analyzer.results
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("=" * 60)
        print(f"Всего листьев: {results['total_leaves']}")
        print(f"Средний процент повреждения: {results['overall_disease_percentage']:.1f}%")
        print(f"Общая здоровая площадь: {results['total_green_area_cm']:.1f} см²")
        print(f"Общая поврежденная площадь: {results['total_damage_area_cm']:.1f} см²")
        print(f"Коэффициент пиксель/см: {results['pixels_per_cm']:.2f}")
        print(f"Фильтрация: min_area = {min_area} пикселей")

        print("\nДетали по листьям:")
        for i, leaf in enumerate(results['leaves']):
            print(f"  Лист {i + 1}: {leaf['disease_percentage']:.1f}% повреждения "
                  f"(здоровая: {leaf['green_area_cm']:.1f} см², "
                  f"поврежденная: {leaf['damage_area_cm']:.1f} см²)")
        analyzer.save_results("leaf_analysis_results.json")



    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
